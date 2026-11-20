// Package scaffold generates OmniAgent services from embedded templates: a
// stdlib Go HTTP service or a FastAPI service, each with tests, a hardened
// Dockerfile and Kubernetes manifest, and a Makefile.
package scaffold

import (
	"bytes"
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"text/template"
	"time"
)

//go:embed all:templates
var templates embed.FS

var Languages = []string{"go", "python"}

// dns1123 is a Kubernetes-compatible name: it becomes the Deployment, Service
// and image name.
var dns1123 = regexp.MustCompile(`^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$`)

type Options struct {
	Name   string
	Lang   string
	Dir    string // parent directory; the service goes in Dir/Name
	Module string // Go module path (default example.com/<name>)
	Port   int
}

type data struct {
	Name, Module string
	Port         int
}

// renamed maps template base names that cannot be dotfiles inside embed.
var renamed = map[string]string{"gitignore": ".gitignore", "dockerignore": ".dockerignore"}

func (o *Options) validate() error {
	if !dns1123.MatchString(o.Name) {
		return fmt.Errorf("service name %q must be lowercase letters, digits and dashes, start with a letter, and be at most 63 characters", o.Name)
	}
	known := false
	for _, l := range Languages {
		known = known || l == o.Lang
	}
	if !known {
		return fmt.Errorf("unknown language %q (want %s)", o.Lang, strings.Join(Languages, " or "))
	}
	if o.Port == 0 {
		o.Port = 8080
	}
	if o.Port < 1024 || o.Port > 65535 {
		return errors.New("port must be between 1024 and 65535 (non-root containers cannot bind lower ports)")
	}
	if o.Module == "" {
		o.Module = "example.com/" + o.Name
	}
	return nil
}

// Generate writes the service and returns the files it created, relative to
// the service directory. It never overwrites an existing directory.
func Generate(o Options) (string, []string, error) {
	if err := o.validate(); err != nil {
		return "", nil, err
	}
	target := filepath.Join(o.Dir, o.Name)
	if _, err := os.Stat(target); err == nil {
		return "", nil, fmt.Errorf("%s already exists", target)
	}
	root := "templates/" + o.Lang
	d := data{Name: o.Name, Module: o.Module, Port: o.Port}
	var created []string
	err := fs.WalkDir(templates, root, func(p string, e fs.DirEntry, err error) error {
		if err != nil || e.IsDir() {
			return err
		}
		raw, err := templates.ReadFile(p)
		if err != nil {
			return err
		}
		tmpl, err := template.New(p).Delims("[[", "]]").Option("missingkey=error").Parse(string(raw))
		if err != nil {
			return err
		}
		var out bytes.Buffer
		if err := tmpl.Execute(&out, d); err != nil {
			return err
		}
		rel := strings.TrimSuffix(strings.TrimPrefix(p, root+"/"), ".tmpl")
		if n, ok := renamed[path.Base(rel)]; ok {
			rel = path.Join(path.Dir(rel), n)
		}
		dest := filepath.Join(target, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			return err
		}
		created = append(created, rel)
		return os.WriteFile(dest, out.Bytes(), 0o644)
	})
	if err != nil {
		os.RemoveAll(target)
		return "", nil, err
	}
	sort.Strings(created)
	return target, created, nil
}

// Step is one verification command and how long it took.
type Step struct {
	Cmd      string        `json:"cmd"`
	Duration time.Duration `json:"duration_ns"`
}

// Verify runs the generated service's own tests: go test for Go; a fresh
// venv, pinned dependency install and pytest for Python.
func Verify(dir, lang string) ([]Step, error) {
	var cmds [][]string
	switch lang {
	case "go":
		cmds = [][]string{{"go", "test", "./..."}}
	case "python":
		py := filepath.Join(".venv", "bin", "python")
		cmds = [][]string{
			{"python3", "-m", "venv", ".venv"},
			{py, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r", "requirements-dev.txt"},
			{py, "-m", "pytest", "-q"},
		}
	default:
		return nil, fmt.Errorf("unknown language %q", lang)
	}
	var steps []Step
	for _, args := range cmds {
		start := time.Now()
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		steps = append(steps, Step{Cmd: strings.Join(args, " "), Duration: time.Since(start)})
		if err != nil {
			return steps, fmt.Errorf("%s failed: %v\n%s", strings.Join(args, " "), err, out)
		}
	}
	return steps, nil
}
