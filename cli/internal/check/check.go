// Package check runs static analysis, dependency audits and security
// compliance rules over a source tree. Built-in rules (secrets, Dockerfiles,
// Kubernetes manifests, dependency pinning) need no external tools; language
// tools (go vet, gofmt, govulncheck, ruff, pip-audit) run when installed.
package check

import (
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

type Severity int

const (
	Info Severity = iota
	Low
	Medium
	High
	Critical
)

var severityNames = []string{"info", "low", "medium", "high", "critical"}

func (s Severity) String() string { return severityNames[s] }

func (s Severity) MarshalText() ([]byte, error) { return []byte(s.String()), nil }

func ParseSeverity(name string) (Severity, error) {
	for i, n := range severityNames {
		if strings.EqualFold(n, name) {
			return Severity(i), nil
		}
	}
	return 0, fmt.Errorf("unknown severity %q (want one of %s)", name, strings.Join(severityNames, ", "))
}

type Finding struct {
	Rule     string   `json:"rule"`
	Severity Severity `json:"severity"`
	File     string   `json:"file"`
	Line     int      `json:"line,omitempty"`
	Message  string   `json:"message"`
	Fix      string   `json:"fix,omitempty"`
}

// ToolRun records an external analyzer: ran, skipped (not installed or not
// applicable), or failed to run.
type ToolRun struct {
	Tool   string `json:"tool"`
	Status string `json:"status"` // ran | skipped | error
	Detail string `json:"detail,omitempty"`
}

type Report struct {
	Root     string    `json:"root"`
	Files    int       `json:"files_scanned"`
	Findings []Finding `json:"findings"`
	Tools    []ToolRun `json:"tools"`
}

// Max returns the highest finding severity, or -1 with no findings.
func (r *Report) Max() Severity {
	max := Severity(-1)
	for _, f := range r.Findings {
		if f.Severity > max {
			max = f.Severity
		}
	}
	return max
}

type Options struct {
	// Fast runs only the built-in rules (no external tools); used by the
	// pre-commit hook.
	Fast bool
}

var skipDirs = map[string]bool{
	".git": true, "node_modules": true, ".venv": true, "venv": true, "vendor": true,
	"__pycache__": true, "dist": true, "build": true, ".pytest_cache": true, ".ruff_cache": true,
}

// file is one scanned file: its path relative to the root and its contents.
type file struct {
	rel  string
	data []byte
}

const maxFileBytes = 2 << 20

// gitFiles lists tracked and untracked-but-not-ignored files when root is in
// a git work tree, so ignored files (local .env, build output) are not
// reported; ok is false outside git.
func gitFiles(root string) (map[string]bool, bool) {
	cmd := exec.Command("git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", ".")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil, false
	}
	keep := map[string]bool{}
	for _, name := range strings.Split(string(out), "\x00") {
		if name != "" {
			keep[filepath.ToSlash(name)] = true
		}
	}
	return keep, true
}

func walk(root string) ([]file, error) {
	tracked, inGit := gitFiles(root)
	var files []file
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if path != root && skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		info, err := d.Info()
		if err != nil || info.Size() > maxFileBytes {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if isBinary(data) {
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		rel = filepath.ToSlash(rel)
		if inGit && !tracked[rel] {
			return nil
		}
		files = append(files, file{rel: rel, data: data})
		return nil
	})
	return files, err
}

func isBinary(data []byte) bool {
	n := len(data)
	if n > 8000 {
		n = 8000
	}
	for _, b := range data[:n] {
		if b == 0 {
			return true
		}
	}
	return false
}

// Run scans root with every rule and returns findings sorted by severity
// (highest first), then file and line.
func Run(root string, opts Options) (*Report, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("%s is not a directory", root)
	}
	files, err := walk(root)
	if err != nil {
		return nil, err
	}
	return scan(root, files, opts), nil
}

// ScanFiles runs the built-in rules over the given contents, keyed by path
// relative to root (e.g. the staged versions of files in a pre-commit hook).
func ScanFiles(root string, contents map[string][]byte) *Report {
	var files []file
	for rel, data := range contents {
		if !isBinary(data) && len(data) <= maxFileBytes {
			files = append(files, file{rel: filepath.ToSlash(rel), data: data})
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].rel < files[j].rel })
	return scan(root, files, Options{Fast: true})
}

func scan(root string, files []file, opts Options) *Report {
	rep := &Report{Root: root, Files: len(files), Findings: []Finding{}, Tools: []ToolRun{}}
	for _, f := range files {
		rep.Findings = append(rep.Findings, scanSecrets(f)...)
		switch {
		case isDockerfile(f.rel):
			rep.Findings = append(rep.Findings, scanDockerfile(f)...)
		case isManifest(f):
			rep.Findings = append(rep.Findings, scanManifest(f)...)
		case isRequirements(f.rel):
			rep.Findings = append(rep.Findings, scanRequirements(f)...)
		}
	}
	if !opts.Fast {
		findings, tools := runTools(root, files)
		rep.Findings = append(rep.Findings, findings...)
		rep.Tools = tools
	}
	sort.SliceStable(rep.Findings, func(i, j int) bool {
		a, b := rep.Findings[i], rep.Findings[j]
		if a.Severity != b.Severity {
			return a.Severity > b.Severity
		}
		if a.File != b.File {
			return a.File < b.File
		}
		return a.Line < b.Line
	})
	return rep
}
