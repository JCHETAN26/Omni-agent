package check

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

func isRequirements(rel string) bool {
	base := path.Base(rel)
	return strings.HasPrefix(base, "requirements") && strings.HasSuffix(base, ".txt")
}

// scanRequirements flags unpinned packages and insecure package indexes.
func scanRequirements(f file) []Finding {
	var out []Finding
	for i, raw := range strings.Split(string(f.data), "\n") {
		line := strings.TrimSpace(strings.SplitN(raw, " #", 2)[0])
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "-") {
			if strings.Contains(line, "index-url") && strings.Contains(line, "http://") {
				out = append(out, Finding{Rule: "deps/insecure-index", Severity: High, File: f.rel, Line: i + 1,
					Message: "package index over plain HTTP", Fix: "use an https:// index"})
			} else if strings.HasPrefix(line, "--extra-index-url") {
				out = append(out, Finding{Rule: "deps/extra-index", Severity: Medium, File: f.rel, Line: i + 1,
					Message: "--extra-index-url lets a public package shadow a private one (dependency confusion)",
					Fix:     "use a single --index-url that proxies both"})
			}
			continue
		}
		if strings.Contains(line, "==") || strings.Contains(line, " @ ") {
			continue
		}
		name := regexp.MustCompile(`^[A-Za-z0-9_.\[\]-]+`).FindString(line)
		out = append(out, Finding{Rule: "deps/unpinned", Severity: Low, File: f.rel, Line: i + 1,
			Message: "dependency " + name + " is not pinned to an exact version",
			Fix:     "pin with == (or a lock file) so builds are reproducible and audits match what ships"})
	}
	return out
}

// findTool looks on PATH, then in Go's bin directory and the project's venv.
func findTool(root, name string) string {
	if p, err := exec.LookPath(name); err == nil {
		return p
	}
	var dirs []string
	if gp := os.Getenv("GOPATH"); gp != "" {
		dirs = append(dirs, filepath.Join(gp, "bin"))
	}
	if home, err := os.UserHomeDir(); err == nil {
		dirs = append(dirs, filepath.Join(home, "go", "bin"))
	}
	dirs = append(dirs, filepath.Join(root, ".venv", "bin"))
	for _, d := range dirs {
		p := filepath.Join(d, name)
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			return p
		}
	}
	return ""
}

const toolTimeout = 5 * time.Minute

func run(dir, bin string, args ...string) (stdout, stderr []byte, err error) {
	ctx, cancel := context.WithTimeout(context.Background(), toolTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Dir = dir
	var o, e bytes.Buffer
	cmd.Stdout, cmd.Stderr = &o, &e
	err = cmd.Run()
	return o.Bytes(), e.Bytes(), err
}

func runTools(root string, files []file) ([]Finding, []ToolRun) {
	var findings []Finding
	var tools []ToolRun
	record := func(tool, status, detail string) { tools = append(tools, ToolRun{tool, status, detail}) }

	var modules, reqs []string
	hasPython := false
	for _, f := range files {
		switch {
		case path.Base(f.rel) == "go.mod":
			modules = append(modules, path.Dir(f.rel))
		case isRequirements(f.rel):
			reqs = append(reqs, f.rel)
		case strings.HasSuffix(f.rel, ".py"):
			hasPython = true
		}
	}
	sort.Strings(modules)

	if len(modules) > 0 {
		gobin := findTool(root, "go")
		if gobin == "" {
			record("go vet", "skipped", "go not installed")
		}
		vuln := findTool(root, "govulncheck")
		for _, mod := range modules {
			dir := filepath.Join(root, mod)
			if gobin != "" {
				fs, tr := goVet(gobin, dir, mod)
				findings, tools = append(findings, fs...), append(tools, tr)
				fs, tr = gofmt(filepath.Join(filepath.Dir(gobin), "gofmt"), dir, mod)
				findings, tools = append(findings, fs...), append(tools, tr)
			}
			if vuln == "" {
				record("govulncheck "+mod, "skipped", "install: go install golang.org/x/vuln/cmd/govulncheck@latest")
				continue
			}
			fs, tr := govulncheck(vuln, dir, mod)
			findings, tools = append(findings, fs...), append(tools, tr)
		}
	}
	if hasPython {
		if ruff := findTool(root, "ruff"); ruff == "" {
			record("ruff", "skipped", "not installed")
		} else {
			fs, tr := ruffCheck(ruff, root)
			findings, tools = append(findings, fs...), append(tools, tr)
		}
	}
	if len(reqs) > 0 {
		audit := findTool(root, "pip-audit")
		for _, req := range reqs {
			if audit == "" {
				record("pip-audit "+req, "skipped", "install: pip install pip-audit")
				continue
			}
			fs, tr := pipAudit(audit, root, req)
			findings, tools = append(findings, fs...), append(tools, tr)
		}
	}
	return findings, tools
}

func rel(mod, p string) string {
	if mod == "." {
		return filepath.ToSlash(p)
	}
	return mod + "/" + filepath.ToSlash(p)
}

var vetLine = regexp.MustCompile(`^(?:\./)?([^:\s]+\.go):(\d+)(?::\d+)?: (.+)$`)

func goVet(gobin, dir, mod string) ([]Finding, ToolRun) {
	_, stderr, err := run(dir, gobin, "vet", "./...")
	var out []Finding
	sc := bufio.NewScanner(bytes.NewReader(stderr))
	for sc.Scan() {
		if m := vetLine.FindStringSubmatch(strings.TrimSpace(sc.Text())); m != nil {
			line, _ := strconv.Atoi(m[2])
			out = append(out, Finding{Rule: "go/vet", Severity: Medium, File: rel(mod, m[1]), Line: line, Message: m[3]})
		}
	}
	if err != nil && len(out) == 0 {
		return nil, ToolRun{"go vet " + mod, "error", firstLine(stderr, err)}
	}
	return out, ToolRun{"go vet " + mod, "ran", fmt.Sprintf("%d findings", len(out))}
}

func gofmt(bin, dir, mod string) ([]Finding, ToolRun) {
	stdout, stderr, err := run(dir, bin, "-l", ".")
	if err != nil {
		return nil, ToolRun{"gofmt " + mod, "error", firstLine(stderr, err)}
	}
	var out []Finding
	for _, p := range strings.Fields(string(stdout)) {
		out = append(out, Finding{Rule: "go/gofmt", Severity: Low, File: rel(mod, p),
			Message: "file is not gofmt-formatted", Fix: "run gofmt -w"})
	}
	return out, ToolRun{"gofmt " + mod, "ran", fmt.Sprintf("%d findings", len(out))}
}

// govulncheck -format json streams osv entries and findings. A finding whose
// trace reaches a function is a vulnerability the code actually calls.
func govulncheck(bin, dir, mod string) ([]Finding, ToolRun) {
	stdout, stderr, err := run(dir, bin, "-format", "json", "./...")
	type frame struct {
		Module   string `json:"module"`
		Function string `json:"function"`
		Position *struct {
			Filename string `json:"filename"`
			Line     int    `json:"line"`
		} `json:"position"`
	}
	type msg struct {
		Config *struct {
			GoVersion string `json:"go_version"`
		} `json:"config"`
		OSV *struct {
			ID      string `json:"id"`
			Summary string `json:"summary"`
		} `json:"osv"`
		Finding *struct {
			OSV          string  `json:"osv"`
			FixedVersion string  `json:"fixed_version"`
			Trace        []frame `json:"trace"`
		} `json:"finding"`
	}
	summaries := map[string]string{}
	goVersion := "the local Go toolchain"
	type hit struct {
		called      bool
		module, fix string
	}
	hits := map[string]*hit{}
	dec := json.NewDecoder(bytes.NewReader(stdout))
	for {
		var m msg
		if err := dec.Decode(&m); err != nil {
			break
		}
		if m.Config != nil && m.Config.GoVersion != "" {
			goVersion = m.Config.GoVersion
		}
		if m.OSV != nil {
			summaries[m.OSV.ID] = m.OSV.Summary
		}
		if m.Finding != nil && len(m.Finding.Trace) > 0 {
			h := hits[m.Finding.OSV]
			if h == nil {
				h = &hit{module: m.Finding.Trace[0].Module, fix: m.Finding.FixedVersion}
				hits[m.Finding.OSV] = h
			}
			h.called = h.called || m.Finding.Trace[0].Function != ""
		}
	}
	if err != nil && len(summaries) == 0 {
		return nil, ToolRun{"govulncheck " + mod, "error", firstLine(stderr, err)}
	}
	var out []Finding
	// Standard-library vulnerabilities come from the toolchain that builds the
	// code, not from go.mod: report them as one upgrade, not one per advisory.
	var stdIDs []string
	stdCalled, stdFix := 0, ""
	for id, h := range hits {
		if h.module != "stdlib" && h.module != "toolchain" {
			continue
		}
		stdIDs = append(stdIDs, id)
		if h.called {
			stdCalled++
		}
		if versionLess(stdFix, h.fix) {
			stdFix = h.fix
		}
		delete(hits, id)
	}
	if len(stdIDs) > 0 {
		sort.Strings(stdIDs)
		sev := Low
		if stdCalled > 0 {
			sev = High
		}
		out = append(out, Finding{Rule: "deps/go-toolchain", Severity: sev, File: rel(mod, "go.mod"),
			Message: fmt.Sprintf("%s has %d known standard-library vulnerabilities (%d reachable from this code): %s",
				goVersion, len(stdIDs), stdCalled, strings.Join(stdIDs, ", ")),
			Fix: "build with Go " + strings.TrimPrefix(stdFix, "v") + " or later (local toolchain and Dockerfile builder image)"})
	}
	for id, h := range hits {
		sev, how := Low, "is in the dependency graph but not called"
		if h.called {
			sev, how = High, "is reachable from this module's code"
		}
		fix := "upgrade " + h.module
		if h.fix != "" {
			fix += " to " + h.fix
		}
		out = append(out, Finding{Rule: "deps/go-vuln", Severity: sev, File: rel(mod, "go.mod"),
			Message: fmt.Sprintf("%s (%s) %s: %s", id, h.module, how, summaries[id]), Fix: fix})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Message < out[j].Message })
	return out, ToolRun{"govulncheck " + mod, "ran", fmt.Sprintf("%d vulnerabilities", len(out))}
}

func ruffCheck(bin, root string) ([]Finding, ToolRun) {
	stdout, stderr, err := run(root, bin, "check", "--output-format", "json", "--exit-zero", ".")
	var items []struct {
		Code     string `json:"code"`
		Message  string `json:"message"`
		Filename string `json:"filename"`
		Location struct {
			Row int `json:"row"`
		} `json:"location"`
	}
	if jerr := json.Unmarshal(stdout, &items); jerr != nil || err != nil {
		return nil, ToolRun{"ruff", "error", firstLine(stderr, errors.Join(err, jerr))}
	}
	var out []Finding
	for _, it := range items {
		sev := Low
		if strings.HasPrefix(it.Code, "S") { // flake8-bandit security rules
			sev = Medium
		}
		r, _ := filepath.Rel(root, it.Filename)
		out = append(out, Finding{Rule: "python/ruff-" + it.Code, Severity: sev, File: filepath.ToSlash(r),
			Line: it.Location.Row, Message: it.Message})
	}
	return out, ToolRun{"ruff", "ran", fmt.Sprintf("%d findings", len(out))}
}

func pipAudit(bin, root, req string) ([]Finding, ToolRun) {
	stdout, stderr, err := run(root, bin, "-r", req, "-f", "json", "--progress-spinner", "off")
	var report struct {
		Dependencies []struct {
			Name  string `json:"name"`
			Ver   string `json:"version"`
			Vulns []struct {
				ID          string   `json:"id"`
				FixVersions []string `json:"fix_versions"`
				Description string   `json:"description"`
			} `json:"vulns"`
		} `json:"dependencies"`
	}
	if jerr := json.Unmarshal(stdout, &report); jerr != nil {
		return nil, ToolRun{"pip-audit " + req, "error", firstLine(stderr, errors.Join(err, jerr))}
	}
	var out []Finding
	for _, d := range report.Dependencies {
		for _, v := range d.Vulns {
			fix := "no fixed version published"
			if len(v.FixVersions) > 0 {
				fix = "upgrade " + d.Name + " to " + strings.Join(v.FixVersions, " or ")
			}
			desc := v.Description
			if len(desc) > 140 {
				desc = desc[:140] + "..."
			}
			out = append(out, Finding{Rule: "deps/python-vuln", Severity: High, File: req,
				Message: fmt.Sprintf("%s in %s %s: %s", v.ID, d.Name, d.Ver, desc), Fix: fix})
		}
	}
	return out, ToolRun{"pip-audit " + req, "ran", fmt.Sprintf("%d vulnerabilities", len(out))}
}

func firstLine(stderr []byte, err error) string {
	s := strings.TrimSpace(string(stderr))
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if s == "" && err != nil {
		s = err.Error()
	}
	return s
}

// versionLess compares "v1.26.9" < "v1.26.11" numerically; "" sorts first.
func versionLess(a, b string) bool {
	pa := strings.Split(strings.TrimPrefix(a, "v"), ".")
	pb := strings.Split(strings.TrimPrefix(b, "v"), ".")
	if a == "" {
		return b != ""
	}
	for i := 0; i < len(pa) || i < len(pb); i++ {
		var x, y int
		if i < len(pa) {
			x, _ = strconv.Atoi(pa[i])
		}
		if i < len(pb) {
			y, _ = strconv.Atoi(pb[i])
		}
		if x != y {
			return x < y
		}
	}
	return false
}
