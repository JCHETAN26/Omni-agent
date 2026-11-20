// Package cmd wires the omni command tree.
package cmd

import (
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"
)

var Version = "dev"

// ExitError carries a non-zero exit code without printing a usage error, e.g.
// `omni check` finding issues at or above --fail-on.
type ExitError struct{ Code int }

func (e *ExitError) Error() string { return fmt.Sprintf("exit status %d", e.Code) }

func NewRoot(stdout, stderr io.Writer) *cobra.Command {
	root := &cobra.Command{
		Use:           "omni",
		Short:         "OmniAgent developer CLI: scaffold services, run checks, and query a gateway",
		Version:       Version,
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.SetOut(stdout)
	root.SetErr(stderr)
	root.AddCommand(newNewCmd(), newCheckCmd(), newHooksCmd(), newConnectCmd(), newSourcesCmd(), newAskCmd())
	return root
}

// Execute runs the CLI and returns the process exit code.
func Execute() int {
	root := NewRoot(os.Stdout, os.Stderr)
	if err := root.Execute(); err != nil {
		var exit *ExitError
		if errors.As(err, &exit) {
			return exit.Code
		}
		fmt.Fprintf(os.Stderr, "omni: %v\n", err)
		return 1
	}
	return 0
}
