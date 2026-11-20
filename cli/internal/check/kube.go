package check

import (
	"bytes"
	"errors"
	"io"
	"strings"

	"gopkg.in/yaml.v3"
)

// isManifest matches rendered Kubernetes YAML. Helm templates ({{ }}) are not
// valid YAML until rendered, so check the rendered manifests instead.
func isManifest(f file) bool {
	if !strings.HasSuffix(f.rel, ".yaml") && !strings.HasSuffix(f.rel, ".yml") {
		return false
	}
	if isHelmTemplate(f) {
		return false
	}
	return bytes.Contains(f.data, []byte("apiVersion:")) && bytes.Contains(f.data, []byte("kind:"))
}

func isHelmTemplate(f file) bool {
	return bytes.Contains(f.data, []byte("{{")) &&
		(strings.HasPrefix(f.rel, "templates/") || strings.Contains(f.rel, "/templates/"))
}

// get walks mapping keys; nil when any step is missing.
func get(n *yaml.Node, keys ...string) *yaml.Node {
	for _, k := range keys {
		if n == nil || n.Kind != yaml.MappingNode {
			return nil
		}
		var next *yaml.Node
		for i := 0; i+1 < len(n.Content); i += 2 {
			if n.Content[i].Value == k {
				next = n.Content[i+1]
				break
			}
		}
		n = next
	}
	return n
}

func isTrue(n *yaml.Node) bool  { return n != nil && n.Value == "true" }
func isFalse(n *yaml.Node) bool { return n != nil && n.Value == "false" }

func podSpec(doc *yaml.Node) (*yaml.Node, string) {
	kind := ""
	if k := get(doc, "kind"); k != nil {
		kind = k.Value
	}
	switch kind {
	case "Pod":
		return get(doc, "spec"), kind
	case "Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "Job", "ReplicationController":
		return get(doc, "spec", "template", "spec"), kind
	case "CronJob":
		return get(doc, "spec", "jobTemplate", "spec", "template", "spec"), kind
	}
	return nil, kind
}

func scanManifest(f file) []Finding {
	var out []Finding
	dec := yaml.NewDecoder(bytes.NewReader(f.data))
	for {
		var doc yaml.Node
		err := dec.Decode(&doc)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil && bytes.Contains(f.data, []byte("{{")) {
			break // an unrendered template: check the rendered output instead
		}
		if err != nil {
			out = append(out, Finding{Rule: "k8s/invalid-yaml", Severity: Low, File: f.rel,
				Message: "could not parse YAML: " + err.Error()})
			break
		}
		if len(doc.Content) == 0 {
			continue
		}
		root := doc.Content[0]
		docs := []*yaml.Node{root}
		if k := get(root, "kind"); k != nil && k.Value == "List" {
			if items := get(root, "items"); items != nil {
				docs = items.Content
			}
		}
		for _, d := range docs {
			out = append(out, scanWorkload(f.rel, d)...)
		}
	}
	return out
}

func scanWorkload(rel string, doc *yaml.Node) []Finding {
	spec, kind := podSpec(doc)
	if spec == nil {
		return nil
	}
	name := kind
	if n := get(doc, "metadata", "name"); n != nil {
		name = kind + "/" + n.Value
	}
	var out []Finding
	add := func(rule string, sev Severity, at *yaml.Node, msg, fix string) {
		line := spec.Line
		if at != nil {
			line = at.Line
		}
		out = append(out, Finding{Rule: rule, Severity: sev, File: rel, Line: line, Message: name + ": " + msg, Fix: fix})
	}
	for _, key := range []string{"hostNetwork", "hostPID", "hostIPC"} {
		if v := get(spec, key); isTrue(v) {
			add("k8s/host-namespace", High, v, key+" shares the node's namespace",
				"remove "+key+" unless the workload is a node agent")
		}
	}
	if vols := get(spec, "volumes"); vols != nil {
		for _, v := range vols.Content {
			if hp := get(v, "hostPath"); hp != nil {
				add("k8s/host-path", Medium, hp, "mounts a hostPath volume", "use a PersistentVolumeClaim, ConfigMap or emptyDir")
			}
		}
	}
	podSec := get(spec, "securityContext")
	podNonRoot := isTrue(get(podSec, "runAsNonRoot"))
	batch := kind == "Job" || kind == "CronJob"
	for _, group := range []string{"initContainers", "containers"} {
		list := get(spec, group)
		if list == nil {
			continue
		}
		for _, c := range list.Content {
			cname := "container"
			if n := get(c, "name"); n != nil {
				cname = "container " + n.Value
			}
			sec := get(c, "securityContext")
			at := sec
			if at == nil {
				at = c
			}
			if v := get(sec, "privileged"); isTrue(v) {
				add("k8s/privileged", Critical, v, cname+" is privileged", "remove privileged: true")
			}
			if !isFalse(get(sec, "allowPrivilegeEscalation")) {
				add("k8s/privilege-escalation", High, at, cname+" allows privilege escalation",
					"set securityContext.allowPrivilegeEscalation: false")
			}
			runAsUser := get(sec, "runAsUser")
			if runAsUser == nil {
				runAsUser = get(podSec, "runAsUser")
			}
			nonRoot := isTrue(get(sec, "runAsNonRoot")) || (podNonRoot && !isFalse(get(sec, "runAsNonRoot")))
			if runAsUser != nil && runAsUser.Value == "0" {
				add("k8s/run-as-root", High, runAsUser, cname+" runs as UID 0", "run as a non-zero UID")
			} else if !nonRoot {
				add("k8s/run-as-root", High, at, cname+" may run as root",
					"set securityContext.runAsNonRoot: true (pod or container level)")
			}
			if !isTrue(get(sec, "readOnlyRootFilesystem")) {
				add("k8s/writable-rootfs", Medium, at, cname+" has a writable root filesystem",
					"set readOnlyRootFilesystem: true and mount an emptyDir for scratch space")
			}
			caps := get(sec, "capabilities")
			dropsAll := false
			if drop := get(caps, "drop"); drop != nil {
				for _, d := range drop.Content {
					dropsAll = dropsAll || strings.EqualFold(d.Value, "ALL")
				}
			}
			if !dropsAll {
				add("k8s/capabilities", Medium, at, cname+" keeps default Linux capabilities",
					"set securityContext.capabilities.drop: [ALL]")
			}
			if addCaps := get(caps, "add"); addCaps != nil {
				for _, a := range addCaps.Content {
					if v := strings.ToUpper(a.Value); v == "SYS_ADMIN" || v == "NET_ADMIN" || v == "ALL" {
						add("k8s/capabilities", High, a, cname+" adds capability "+v, "remove it")
					}
				}
			}
			res := get(c, "resources")
			if get(res, "limits", "memory") == nil {
				add("k8s/no-resource-limits", Medium, c, cname+" has no memory limit",
					"set resources.limits.memory so one pod cannot exhaust a node")
			}
			if get(res, "requests", "cpu") == nil {
				add("k8s/no-resource-requests", Low, c, cname+" has no CPU request",
					"set resources.requests.cpu so the scheduler can place it")
			}
			if img := get(c, "image"); img != nil {
				name := img.Value[strings.LastIndex(img.Value, "/")+1:]
				if !strings.Contains(img.Value, "@sha256:") && (!strings.Contains(name, ":") || strings.HasSuffix(name, ":latest")) {
					add("k8s/image-latest", Medium, img, cname+" image "+img.Value+" is not pinned",
						"use a version tag or digest so rollouts are reproducible")
				}
			}
			if group == "containers" && !batch && get(c, "readinessProbe") == nil {
				add("k8s/no-readiness-probe", Low, c, cname+" has no readinessProbe",
					"add a readinessProbe so traffic waits until the pod can serve")
			}
		}
	}
	return out
}
