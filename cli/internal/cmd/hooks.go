package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

const hookMarker = "# installed by omni hooks install"

const hookScript = `#!/bin/sh
` + hookMarker + `
# Blocks the commit when staged files have high or critical findings.
# Bypass once with: git commit --no-verify
exec omni check --staged --fail-on high
`

func newHooksCmd() *cobra.Command {
	hooks := &cobra.Command{Use: "hooks", Short: "Manage git hooks"}
	var force bool
	install := &cobra.Command{
		Use:   "install [repo]",
		Short: "Install a pre-commit hook that runs omni check on staged files",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			dir := "."
			if len(args) == 1 {
				dir = args[0]
			}
			hooksDir, err := git(dir, "rev-parse", "--git-path", "hooks")
			if err != nil {
				return fmt.Errorf("not a git repository: %w", err)
			}
			hooksDir = strings.TrimSpace(hooksDir)
			if !filepath.IsAbs(hooksDir) {
				hooksDir = filepath.Join(dir, hooksDir)
			}
			path := filepath.Join(hooksDir, "pre-commit")
			if existing, err := os.ReadFile(path); err == nil && !strings.Contains(string(existing), hookMarker) && !force {
				return fmt.Errorf("%s already exists and was not installed by omni; rerun with --force to replace it", path)
			}
			if err := os.MkdirAll(hooksDir, 0o755); err != nil {
				return err
			}
			if err := os.WriteFile(path, []byte(hookScript), 0o755); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "installed %s\n", path)
			return nil
		},
	}
	install.Flags().BoolVar(&force, "force", false, "replace an existing pre-commit hook")
	hooks.AddCommand(install)
	return hooks
}
