package scaffold

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/JCHETAN26/Omni-agent/cli/internal/check"
)

func TestValidation(t *testing.T) {
	for _, o := range []Options{
		{Name: "Bad_Name", Lang: "go"},
		{Name: "1svc", Lang: "go"},
		{Name: "svc-", Lang: "go"},
		{Name: "svc", Lang: "rust"},
		{Name: "svc", Lang: "go", Port: 80},
	} {
		o.Dir = t.TempDir()
		if _, _, err := Generate(o); err == nil {
			t.Errorf("%+v accepted", o)
		}
	}
}

func TestRefusesToOverwrite(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "svc"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Generate(Options{Name: "svc", Lang: "go", Dir: dir}); err == nil ||
		!strings.Contains(err.Error(), "already exists") {
		t.Fatalf("want already-exists error, got %v", err)
	}
}

// Every generated service must pass omni check with zero findings: the
// scaffold is the reference for what the checks expect.
func TestGeneratedServicesPassChecks(t *testing.T) {
	for _, lang := range Languages {
		t.Run(lang, func(t *testing.T) {
			target, files, err := Generate(Options{Name: "incident-notes", Lang: lang, Dir: t.TempDir(), Port: 9090})
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range []string{"Dockerfile", "Makefile", "README.md", ".gitignore", ".dockerignore", "deploy/k8s.yaml"} {
				if !contains(files, want) {
					t.Errorf("missing %s in %v", want, files)
				}
			}
			for _, f := range files {
				raw, _ := os.ReadFile(filepath.Join(target, f))
				if strings.Contains(string(raw), "[[") {
					t.Errorf("%s has an unrendered template marker", f)
				}
				if !strings.Contains(string(raw), "incident-notes") && (f == "Dockerfile" || f == "deploy/k8s.yaml") {
					t.Errorf("%s does not use the service name", f)
				}
			}
			rep, err := check.Run(target, check.Options{Fast: lang == "python"})
			if err != nil {
				t.Fatal(err)
			}
			// deps/go-toolchain reflects the Go installed on this machine, not
			// the generated code.
			var own []check.Finding
			for _, f := range rep.Findings {
				if f.Rule != "deps/go-toolchain" {
					own = append(own, f)
				}
			}
			if len(own) != 0 {
				t.Fatalf("generated %s service has findings: %+v", lang, own)
			}
		})
	}
}

func TestGeneratedGoServiceTestsPass(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not installed")
	}
	target, _, err := Generate(Options{Name: "incident-notes", Lang: "go", Dir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Verify(target, "go"); err != nil {
		t.Fatal(err)
	}
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}
