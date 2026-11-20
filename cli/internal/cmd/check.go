package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/JCHETAN26/Omni-agent/cli/internal/check"
	"github.com/spf13/cobra"
)

func newCheckCmd() *cobra.Command {
	var (
		asJSON, fast, staged bool
		failOn               string
	)
	c := &cobra.Command{
		Use:   "check [dir]",
		Short: "Static analysis, dependency audit and security checks",
		Long: `Scan a source tree for leaked secrets, insecure Dockerfiles and Kubernetes
manifests, unpinned or vulnerable dependencies, and Go/Python lint issues.

Built-in rules always run. go vet, gofmt, govulncheck, ruff and pip-audit run
when installed; the report lists any that were skipped.`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			threshold, err := check.ParseSeverity(failOn)
			if err != nil {
				return err
			}
			dir := "."
			if len(args) == 1 {
				dir = args[0]
			}
			var rep *check.Report
			if staged {
				rep, err = checkStaged(dir)
			} else {
				rep, err = check.Run(dir, check.Options{Fast: fast})
			}
			if err != nil {
				return err
			}
			if asJSON {
				enc := json.NewEncoder(cmd.OutOrStdout())
				enc.SetIndent("", "  ")
				if err := enc.Encode(rep); err != nil {
					return err
				}
			} else {
				printReport(cmd.OutOrStdout(), rep, threshold)
			}
			if rep.Max() >= threshold {
				return &ExitError{Code: 1}
			}
			return nil
		},
	}
	c.Flags().BoolVar(&asJSON, "json", false, "print the report as JSON")
	c.Flags().BoolVar(&fast, "fast", false, "built-in rules only (skip external tools)")
	c.Flags().BoolVar(&staged, "staged", false, "check only files staged in git (used by the pre-commit hook)")
	c.Flags().StringVar(&failOn, "fail-on", "high", "exit 1 when a finding is at least this severity (info|low|medium|high|critical)")
	return c
}

// checkStaged scans the index versions of added, copied and modified files,
// so a hook checks exactly what is about to be committed.
func checkStaged(dir string) (*check.Report, error) {
	top, err := git(dir, "rev-parse", "--show-toplevel")
	if err != nil {
		return nil, fmt.Errorf("--staged needs a git repository: %w", err)
	}
	top = strings.TrimSpace(top)
	names, err := git(top, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
	if err != nil {
		return nil, err
	}
	contents := map[string][]byte{}
	for _, name := range strings.Split(names, "\x00") {
		if name == "" {
			continue
		}
		data, err := git(top, "show", ":"+name)
		if err != nil {
			return nil, err
		}
		contents[name] = []byte(data)
	}
	return check.ScanFiles(top, contents), nil
}

func git(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("git %s: %v %s", strings.Join(args, " "), err, strings.TrimSpace(errb.String()))
	}
	return out.String(), nil
}

func printReport(w io.Writer, rep *check.Report, threshold check.Severity) {
	counts := map[check.Severity]int{}
	for _, f := range rep.Findings {
		counts[f.Severity]++
		loc := f.File
		if f.Line > 0 {
			loc = fmt.Sprintf("%s:%d", f.File, f.Line)
		}
		mark := " "
		if f.Severity >= threshold {
			mark = "✗"
		}
		fmt.Fprintf(w, "%s %-8s %-28s %s\n           %s\n", mark, strings.ToUpper(f.Severity.String()), f.Rule, loc, f.Message)
		if f.Fix != "" {
			fmt.Fprintf(w, "           fix: %s\n", f.Fix)
		}
	}
	var parts []string
	for s := check.Critical; s >= check.Info; s-- {
		if counts[s] > 0 {
			parts = append(parts, fmt.Sprintf("%d %s", counts[s], s))
		}
	}
	summary := "no findings"
	if len(parts) > 0 {
		summary = strings.Join(parts, ", ")
	}
	root, _ := filepath.Abs(rep.Root)
	fmt.Fprintf(w, "\n%s: %d files scanned, %s (fail-on: %s)\n", root, rep.Files, summary, threshold)
	for _, t := range rep.Tools {
		line := fmt.Sprintf("  %-30s %s", t.Tool, t.Status)
		if t.Detail != "" {
			line += " — " + t.Detail
		}
		fmt.Fprintln(w, line)
	}
}
