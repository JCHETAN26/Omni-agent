"""Render the Helm chart and check structural invariants (needs the `helm` CLI)."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CHART = ROOT / "deploy/kubernetes/helm/omniagent"
MANIFESTS = ROOT / "deploy/kubernetes/manifests/omniagent.yaml"

pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")

EXTERNAL = [
    "qdrant.enabled=false",
    "qdrant.externalUrl=https://q.example:6333",
    "elasticsearch.enabled=false",
    "elasticsearch.externalUrl=https://es.example:9243",
    "redis.enabled=false",
    "redis.externalUrl=rediss://:pw@r.example:6380/0",
    "secrets.existingSecret=omni-prod",
    "config.llmProvider=anthropic",
]


def render(*sets: str, skip_tests: bool = True) -> list[dict]:
    cmd = ["helm", "template", "omniagent", str(CHART), "--namespace", "omniagent"]
    if skip_tests:
        cmd.append("--skip-tests")
    for s in sets:
        cmd += ["--set", s]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def by_kind(docs, kind):
    return {d["metadata"]["name"]: d for d in docs if d["kind"] == kind}


def pod_specs(docs):
    for d in docs:
        if d["kind"] in ("Deployment", "StatefulSet", "Job"):
            yield d, d["spec"]["template"]


def test_default_render_has_every_component():
    docs = render()
    assert set(by_kind(docs, "Deployment")) == {"omniagent-backend", "omniagent-frontend"}
    assert set(by_kind(docs, "StatefulSet")) == {
        "omniagent-qdrant",
        "omniagent-elasticsearch",
        "omniagent-redis",
    }
    assert "omniagent-seed" in by_kind(docs, "Job")
    assert len(by_kind(docs, "NetworkPolicy")) == 3


def test_selectors_match_pod_labels_and_services_resolve():
    docs = render()
    pods = {}
    for d, tpl in pod_specs(docs):
        labels = tpl["metadata"]["labels"]
        sel = d["spec"].get("selector", {}).get("matchLabels")
        if sel:
            assert sel.items() <= labels.items(), d["metadata"]["name"]
        pods[d["metadata"]["name"]] = (labels, tpl["spec"])
    for name, svc in by_kind(docs, "Service").items():
        sel = svc["spec"]["selector"]
        matches = [p for p, (labels, _) in pods.items() if sel.items() <= labels.items()]
        assert matches, f"service {name} selects no pods"
        port_names = {
            p["name"]
            for _, (labels, spec) in pods.items()
            if sel.items() <= labels.items()
            for c in spec["containers"]
            for p in c.get("ports", [])
        }
        for port in svc["spec"]["ports"]:
            target = port.get("targetPort")
            if isinstance(target, str):
                assert target in port_names, f"{name}: targetPort {target} not on any pod"


def test_workloads_are_hardened_and_bounded():
    for d, tpl in pod_specs(render()):
        spec = tpl["spec"]
        name = d["metadata"]["name"]
        assert spec["securityContext"]["runAsNonRoot"] is True, name
        for c in spec["containers"]:
            assert c["resources"]["limits"] and c["resources"]["requests"], name
            assert c["securityContext"]["allowPrivilegeEscalation"] is False, name
            assert c["securityContext"]["capabilities"]["drop"] == ["ALL"], name
            assert ":" in c["image"] and not c["image"].endswith(":latest"), name
            if d["kind"] != "Job":
                assert "readinessProbe" in c and "livenessProbe" in c, name


def test_backend_wiring_and_probes():
    docs = render()
    cm = by_kind(docs, "ConfigMap")["omniagent-config"]["data"]
    assert cm["QDRANT_URL"] == "http://omniagent-qdrant:6333"
    assert cm["ELASTICSEARCH_URL"] == "http://omniagent-elasticsearch:9200"
    assert cm["REDIS_URL"] == "redis://omniagent-redis:6379/0"
    api = by_kind(docs, "Deployment")["omniagent-backend"]["spec"]["template"]
    c = api["spec"]["containers"][0]
    assert c["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert c["livenessProbe"]["httpGet"]["path"] == "/livez"
    assert {e["secretRef"]["name"] for e in c["envFrom"] if "secretRef" in e} == {
        "omniagent-secrets"
    }
    assert api["metadata"]["annotations"]["prometheus.io/path"] == "/metrics"


def test_ingress_exposes_only_gateway_paths_with_websocket_timeouts():
    ing = by_kind(render(), "Ingress")["omniagent"]
    paths = {
        p["path"]: p["backend"]["service"]["name"] for p in ing["spec"]["rules"][0]["http"]["paths"]
    }
    assert paths == {
        "/ws": "omniagent-backend",
        "/sse": "omniagent-backend",
        "/health": "omniagent-backend",
        "/": "omniagent-frontend",
    }
    ann = ing["metadata"]["annotations"]
    assert int(ann["nginx.ingress.kubernetes.io/proxy-read-timeout"]) >= 3600


def test_external_services_and_existing_secret():
    docs = render(*EXTERNAL)
    assert not by_kind(docs, "StatefulSet") and not by_kind(docs, "NetworkPolicy")
    assert not by_kind(docs, "Secret")  # existingSecret is used instead
    cm = by_kind(docs, "ConfigMap")["omniagent-config"]["data"]
    assert cm["QDRANT_URL"] == "https://q.example:6333"
    assert "REDIS_URL" not in cm  # credentials never land in the ConfigMap
    assert cm["LLM_PROVIDER"] == "anthropic"


def test_redis_url_with_password_goes_to_the_secret():
    docs = render("redis.enabled=false", "redis.externalUrl=rediss://:pw@r.example:6380/0")
    secret = by_kind(docs, "Secret")["omniagent-secrets"]["stringData"]
    assert secret["REDIS_URL"] == "rediss://:pw@r.example:6380/0"


def test_optional_features_render():
    docs = render(
        "backend.autoscaling.enabled=true",
        "metrics.serviceMonitor.enabled=true",
        "ingress.tls.enabled=true",
        "triton.enabled=true",
        "triton.image.tag=25.01-py3",
        "triton.modelRepositoryClaim=models",
        "triton.gpu=true",
        "config.inferenceBackend=triton",
        "config.embeddingProvider=triton",
    )
    assert "replicas" not in by_kind(docs, "Deployment")["omniagent-backend"]["spec"]
    assert by_kind(docs, "HorizontalPodAutoscaler")
    assert by_kind(docs, "ServiceMonitor")
    triton = by_kind(docs, "Deployment")["omniagent-triton"]["spec"]["template"]["spec"]
    assert triton["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert by_kind(docs, "ConfigMap")["omniagent-config"]["data"]["TRITON_GRPC_URL"] == (
        "omniagent-triton:8001"
    )


@pytest.mark.parametrize(
    "sets,message",
    [
        (["triton.enabled=true"], "triton.image.tag is required"),
        (["config.inferenceBackend=triton"], "requires triton.enabled"),
        (["config.visionBackend=triton"], "requires triton.enabled"),
    ],
)
def test_misconfiguration_fails_fast(sets, message):
    with pytest.raises(subprocess.CalledProcessError) as info:
        render(*sets)
    assert message in info.value.stderr


def test_grafana_dashboard_is_valid_and_uses_exported_metrics():
    import json

    from backend.core import metrics

    cm = by_kind(render(), "ConfigMap")["omniagent-dashboard"]
    dash = json.loads(cm["data"]["omniagent.json"])
    exported = {m._name for m in vars(metrics).values() if hasattr(m, "_name")}
    used = {e["expr"] for p in dash["panels"] for e in p["targets"]}
    for name in exported:
        base = name.removesuffix("_total")
        assert any(base in expr for expr in used), f"metric {name} not on the dashboard"


def test_committed_manifests_match_the_chart():
    rendered = render()
    committed = [d for d in yaml.safe_load_all(MANIFESTS.read_text()) if d]
    assert committed == rendered, "run deploy/kubernetes/render.sh"


def test_triton_with_baked_models_runs_hardened_without_a_pvc():
    docs = render(
        "triton.enabled=true",
        "triton.image.repository=omniagent-triton",
        "triton.image.tag=dev",
        "config.visionBackend=triton",
    )
    spec = by_kind(docs, "Deployment")["omniagent-triton"]["spec"]["template"]["spec"]
    assert {v["name"] for v in spec["volumes"]} == {"shm", "tmp"}
    container = spec["containers"][0]
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert spec["securityContext"]["runAsNonRoot"] is True
    config = by_kind(docs, "ConfigMap")["omniagent-config"]["data"]
    assert (
        config["VISION_BACKEND"] == "triton"
        and config["TRITON_GRPC_URL"] == "omniagent-triton:8001"
    )
    policies = by_kind(docs, "NetworkPolicy")
    assert policies["omniagent-triton"]["spec"]["ingress"][0]["ports"][0]["port"] == 8001


def test_qdrant_root_filesystem_is_read_only():
    qdrant = by_kind(render(), "StatefulSet")["omniagent-qdrant"]["spec"]["template"]["spec"]
    container = qdrant["containers"][0]
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    env = {e["name"]: e["value"] for e in container["env"]}
    mounts = {m["mountPath"] for m in container["volumeMounts"]}
    assert env["QDRANT_INIT_FILE_PATH"].startswith("/qdrant/init/") and "/qdrant/init" in mounts
