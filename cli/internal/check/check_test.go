package check

import (
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// Fake credentials are assembled at runtime so this file never contains one.
var (
	fakeAWS    = "AKIA" + "Q3ZK7MW2RT9PLX4N"
	fakeGitHub = "ghp_" + strings.Repeat("a1B2c3D4", 5)
	fakeOpenAI = "sk-proj-" + "Zx9Qw8Er7Ty6Ui5Op4As3Df2Gh1Jk0LmNb"
	pemHeader  = "-----BEGIN " + "RSA PRIVATE KEY-----"
	fakePass   = "k9#Lm2$v" + "Q8pZ!wX4"
)

func writeTree(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for name, body := range files {
		p := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func rules(t *testing.T, files map[string]string) map[string]int {
	t.Helper()
	rep, err := Run(writeTree(t, files), Options{Fast: true})
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]int{}
	for _, f := range rep.Findings {
		got[f.Rule]++
	}
	return got
}

func keys(m map[string]int) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func TestSecrets(t *testing.T) {
	got := rules(t, map[string]string{
		"aws.py":       "key = '" + fakeAWS + "'\n",
		"gh.sh":        "export GH=" + fakeGitHub + "\n",
		"llm.py":       "client = OpenAI(api_key='" + fakeOpenAI + "')\n",
		"id_rsa":       pemHeader + "\nMIIE\n",
		"settings.py":  "DB_PASSWORD = '" + fakePass + "'\n",
		".env":         "LLM_API_KEY=abc123\n",
		".env.example": "LLM_API_KEY=\n",
	})
	want := map[string]int{
		"secret/aws-access-key": 1, "secret/github-token": 1, "secret/openai-key": 1,
		"secret/private-key": 1, "secret/hardcoded-credential": 1, "secret/env-file": 1,
	}
	for rule, n := range want {
		if got[rule] != n {
			t.Errorf("%s: want %d, got %d (all: %v)", rule, n, got[rule], got)
		}
	}
}

func TestSecretsIgnoresPlaceholdersAndAllowMarker(t *testing.T) {
	got := rules(t, map[string]string{
		"a.py": "api_key = 'your-api-key-here'\npassword = '${DB_PASSWORD}'\n",
		"b.py": "token = '" + fakeGitHub + "'  # omni:allow-secret test fixture\n",
		"c.py": "token = os.environ['TOKEN']\nsecret_name = 'short'\n",
		"d.md": "Set LLM_API_KEY in your environment.\n",
	})
	if len(got) != 0 {
		t.Fatalf("false positives: %v", got)
	}
}

func TestDockerfileRules(t *testing.T) {
	bad := `FROM python
ENV OPENAI_API_KEY=abc123
ADD https://example.com/tool.tgz /opt/
RUN curl -fsSL https://get.example.sh | \
    bash
RUN apt-get update && sudo apt-get install -y git
USER root
`
	got := rules(t, map[string]string{"Dockerfile": bad})
	for _, rule := range []string{"docker/unpinned-base", "docker/secret-in-image", "docker/add-remote",
		"docker/curl-pipe-shell", "docker/sudo", "docker/root-user", "docker/no-healthcheck"} {
		if got[rule] == 0 {
			t.Errorf("%s not reported (got %v)", rule, keys(got))
		}
	}
}

func TestDockerfileCleanMultiStage(t *testing.T) {
	good := `# build
FROM golang:1.25-bookworm AS build
RUN go build -o /out/app .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/app /app
HEALTHCHECK CMD ["/app", "-healthcheck"]
ENTRYPOINT ["/app"]
`
	if got := rules(t, map[string]string{"Dockerfile": good}); len(got) != 0 {
		t.Fatalf("clean Dockerfile flagged: %v", got)
	}
	rootless := "FROM python:3.12-slim\nRUN useradd -u 10001 app\nUSER 10001\nHEALTHCHECK CMD true\n"
	if got := rules(t, map[string]string{"svc.Dockerfile": rootless}); len(got) != 0 {
		t.Fatalf("clean Dockerfile flagged: %v", got)
	}
}

const badDeployment = `apiVersion: v1
kind: ConfigMap
metadata: {name: cfg}
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: api}
spec:
  template:
    spec:
      hostNetwork: true
      volumes:
        - name: docker
          hostPath: {path: /var/run/docker.sock}
      containers:
        - name: api
          image: example/api:latest
          securityContext:
            privileged: true
            runAsUser: 0
            capabilities: {add: [SYS_ADMIN]}
`

const goodDeployment = `apiVersion: apps/v1
kind: Deployment
metadata: {name: api}
spec:
  template:
    spec:
      securityContext: {runAsNonRoot: true}
      containers:
        - name: api
          image: example/api:1.4.2
          readinessProbe: {httpGet: {path: /readyz, port: 8080}}
          resources:
            requests: {cpu: 100m, memory: 128Mi}
            limits: {memory: 256Mi}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: {drop: [ALL]}
---
apiVersion: batch/v1
kind: CronJob
metadata: {name: reindex}
spec:
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: reindex
              image: example/api@sha256:0123456789abcdef
              resources: {requests: {cpu: 50m}, limits: {memory: 128Mi}}
              securityContext:
                runAsNonRoot: true
                allowPrivilegeEscalation: false
                readOnlyRootFilesystem: true
                capabilities: {drop: [ALL]}
`

func TestKubernetesRules(t *testing.T) {
	got := rules(t, map[string]string{"deploy/api.yaml": badDeployment})
	for _, rule := range []string{"k8s/host-namespace", "k8s/host-path", "k8s/privileged", "k8s/privilege-escalation",
		"k8s/run-as-root", "k8s/writable-rootfs", "k8s/capabilities", "k8s/no-resource-limits",
		"k8s/no-resource-requests", "k8s/image-latest", "k8s/no-readiness-probe"} {
		if got[rule] == 0 {
			t.Errorf("%s not reported (got %v)", rule, keys(got))
		}
	}
	if got := rules(t, map[string]string{"deploy/api.yaml": goodDeployment}); len(got) != 0 {
		t.Fatalf("hardened manifests flagged: %v", got)
	}
	helm := "apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: {{ .Values.replicas }}\n"
	if got := rules(t, map[string]string{"chart/templates/d.yaml": helm}); len(got) != 0 {
		t.Fatalf("Helm template should be skipped: %v", got)
	}
}

func TestRequirementsRules(t *testing.T) {
	got := rules(t, map[string]string{"requirements.txt": strings.Join([]string{
		"# comment", "fastapi==0.115.6", "uvicorn[standard]>=0.30", "requests",
		"--extra-index-url https://pypi.internal/simple",
		"--index-url http://mirror.local/simple",
		"mypkg @ https://example.com/mypkg-1.0.tar.gz",
	}, "\n")})
	if got["deps/unpinned"] != 2 || got["deps/extra-index"] != 1 || got["deps/insecure-index"] != 1 {
		t.Fatalf("unexpected: %v", got)
	}
}

func TestSkipsVendoredAndBinaryFiles(t *testing.T) {
	got := rules(t, map[string]string{
		"node_modules/x/key.js": "k='" + fakeAWS + "'",
		".venv/lib/a.py":        "k='" + fakeAWS + "'",
		"blob.bin":              "\x00\x01" + fakeAWS,
	})
	if len(got) != 0 {
		t.Fatalf("scanned skipped paths: %v", got)
	}
}

func TestReportOrderAndMax(t *testing.T) {
	rep, err := Run(writeTree(t, map[string]string{
		"Dockerfile": "FROM python:3.12-slim\nUSER 1000\n", // low: no healthcheck
		"a.py":       "key = '" + fakeAWS + "'\n",          // critical
	}), Options{Fast: true})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Max() != Critical || rep.Findings[0].Severity != Critical ||
		rep.Findings[len(rep.Findings)-1].Severity != Low {
		t.Fatalf("bad ordering: %+v", rep.Findings)
	}
	empty, _ := Run(t.TempDir(), Options{Fast: true})
	if empty.Max() != -1 {
		t.Fatal("empty tree should have no max severity")
	}
}

func TestGoVetIntegration(t *testing.T) {
	if findTool("", "go") == "" {
		t.Skip("go not installed")
	}
	root := writeTree(t, map[string]string{
		"svc/go.mod":  "module example.com/svc\n\ngo 1.22\n",
		"svc/main.go": "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Printf(\"%d\\n\", \"x\")\n}\n",
		"svc/ugly.go": "package main\nfunc   helper() {}\n",
	})
	rep, err := Run(root, Options{})
	if err != nil {
		t.Fatal(err)
	}
	var vet, fmtd bool
	for _, f := range rep.Findings {
		vet = vet || (f.Rule == "go/vet" && f.File == "svc/main.go" && f.Line == 6)
		fmtd = fmtd || (f.Rule == "go/gofmt" && f.File == "svc/ugly.go")
	}
	if !vet || !fmtd {
		t.Fatalf("go vet/gofmt findings missing: %+v tools=%+v", rep.Findings, rep.Tools)
	}
}

func TestRegressions(t *testing.T) {
	// Non-root-by-default base images.
	if got := rules(t, map[string]string{"Dockerfile": "FROM nginxinc/nginx-unprivileged:1.30.6-alpine\nHEALTHCHECK CMD true\n"}); len(got) != 0 {
		t.Errorf("nginx-unprivileged flagged: %v", got)
	}
	// {{ }} inside a valid rendered manifest (a Grafana dashboard) must not hide it.
	rendered := "apiVersion: v1\nkind: ConfigMap\ndata:\n  d.json: '{\"legendFormat\": \"{{outcome}}\"}'\n---\n" + badDeployment
	if got := rules(t, map[string]string{"manifests/all.yaml": rendered}); got["k8s/privileged"] == 0 {
		t.Errorf("rendered manifest with {{ }} was skipped: %v", got)
	}
	// An unrendered template outside templates/ that fails to parse is skipped, not reported.
	tmpl := "apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: {{ .Values.n }}\n  x: {{- if .Values.y }}\n"
	if got := rules(t, map[string]string{"k8s/d.yaml": tmpl}); len(got) != 0 {
		t.Errorf("unrendered template reported: %v", got)
	}
}

func TestRespectsGitignore(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	root := writeTree(t, map[string]string{
		".gitignore": ".env\n",
		".env":       "LLM_API_KEY=" + fakeOpenAI + "\n",
		"app.py":     "key = '" + fakeAWS + "'\n",
	})
	if out, err := exec.Command("git", "-C", root, "init", "-q").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v %s", err, out)
	}
	rep, err := Run(root, Options{Fast: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range rep.Findings {
		if f.File == ".env" {
			t.Errorf("ignored file reported: %+v", f)
		}
	}
	if len(rep.Findings) != 1 || rep.Findings[0].File != "app.py" {
		t.Errorf("want only the untracked, unignored app.py finding: %+v", rep.Findings)
	}
}

func TestVersionLess(t *testing.T) {
	for _, c := range []struct {
		a, b string
		want bool
	}{{"v1.26.9", "v1.26.11", true}, {"v1.26.11", "v1.26.9", false}, {"", "v1.0.0", true}, {"v1.2.3", "v1.2.3", false}} {
		if got := versionLess(c.a, c.b); got != c.want {
			t.Errorf("versionLess(%q, %q) = %v", c.a, c.b, got)
		}
	}
}
