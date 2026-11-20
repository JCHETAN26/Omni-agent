package cmd

import (
	"fmt"
	"strings"
	"time"

	"github.com/JCHETAN26/Omni-agent/cli/internal/check"
	"github.com/JCHETAN26/Omni-agent/cli/internal/scaffold"
	"github.com/spf13/cobra"
)

func newNewCmd() *cobra.Command {
	newCmd := &cobra.Command{Use: "new", Short: "Generate new components"}
	var (
		o              scaffold.Options
		verify, noChck bool
	)
	svc := &cobra.Command{
		Use:   "service <name>",
		Short: "Generate a service with tests, hardened Dockerfile and Kubernetes manifest",
		Long: `Generate a service that passes omni check out of the box: health and
readiness endpoints, graceful SIGTERM draining, Prometheus metrics, tests, a
non-root Dockerfile with a HEALTHCHECK, and a Kubernetes manifest with probes,
resource limits, a read-only root filesystem and all capabilities dropped.

With --verify it also runs the new service's tests (for Python: a fresh venv
and a pinned dependency install first).`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			out := cmd.OutOrStdout()
			start := time.Now()
			o.Name = args[0]
			target, files, err := scaffold.Generate(o)
			if err != nil {
				return err
			}
			fmt.Fprintf(out, "created %s (%s, %d files)\n", target, o.Lang, len(files))
			for _, f := range files {
				fmt.Fprintf(out, "  %s\n", f)
			}
			// Findings are reported but do not skip --verify; they decide the
			// exit code at the end.
			blocking := false
			if !noChck {
				rep, err := check.Run(target, check.Options{})
				if err != nil {
					return err
				}
				fmt.Fprintf(out, "\nomni check: ")
				if len(rep.Findings) == 0 {
					fmt.Fprintln(out, "no findings")
					for _, t := range rep.Tools {
						fmt.Fprintf(out, "  %-30s %s\n", t.Tool, t.Status)
					}
				} else {
					fmt.Fprintln(out)
					printReport(out, rep, check.High)
					blocking = rep.Max() >= check.High
				}
			}
			if verify {
				steps, err := scaffold.Verify(target, o.Lang)
				for _, s := range steps {
					fmt.Fprintf(out, "  %-60s %6.1fs\n", truncate(s.Cmd, 60), s.Duration.Seconds())
				}
				if err != nil {
					return err
				}
				fmt.Fprintln(out, "tests: passed")
			}
			fmt.Fprintf(out, "\ndone in %.1fs. Next: cd %s && make test\n", time.Since(start).Seconds(), target)
			if blocking {
				fmt.Fprintln(out, "omni check reported high or critical findings (see above)")
				return &ExitError{Code: 1}
			}
			return nil
		},
	}
	svc.Flags().StringVar(&o.Lang, "lang", "go", "service language: "+strings.Join(scaffold.Languages, " | "))
	svc.Flags().StringVar(&o.Dir, "dir", ".", "parent directory for the new service")
	svc.Flags().StringVar(&o.Module, "module", "", "Go module path (default example.com/<name>)")
	svc.Flags().IntVar(&o.Port, "port", 8080, "port the service listens on")
	svc.Flags().BoolVar(&verify, "verify", false, "run the generated service's tests")
	svc.Flags().BoolVar(&noChck, "no-check", false, "skip omni check on the generated service")
	newCmd.AddCommand(svc)
	return newCmd
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
