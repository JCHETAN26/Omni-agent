package cmd

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/JCHETAN26/Omni-agent/cli/internal/client"
	"github.com/JCHETAN26/Omni-agent/cli/internal/config"
	"github.com/spf13/cobra"
)

func loadState() (string, config.State, error) {
	path, err := config.Path()
	if err != nil {
		return "", config.State{}, err
	}
	s, err := config.Load(path)
	return path, s, err
}

func connected() (string, config.State, error) {
	path, s, err := loadState()
	if err == nil && s.Server == "" {
		err = errors.New("connect to a gateway first with omni connect --server URL")
	}
	return path, s, err
}

func newConnectCmd() *cobra.Command {
	var server string
	c := &cobra.Command{
		Use:   "connect",
		Short: "Point the CLI at an OmniAgent gateway",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			server, err := client.ValidateServer(server)
			if err != nil {
				return err
			}
			path, s, err := loadState()
			if err != nil {
				return err
			}
			ready, err := client.New(server).Ready(cmd.Context())
			if err != nil {
				return err
			}
			if !ready {
				return errors.New("gateway is not ready")
			}
			if server != s.Server {
				s = config.State{Server: server} // a new gateway has no thread to continue
			}
			if err := config.Save(path, s); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "Connected to %s\n", server)
			return nil
		},
	}
	c.Flags().StringVar(&server, "server", "", "gateway URL, e.g. http://127.0.0.1:8000")
	_ = c.MarkFlagRequired("server")
	return c
}

func newSourcesCmd() *cobra.Command {
	sources := &cobra.Command{Use: "sources", Short: "Manage knowledge sources"}
	sources.AddCommand(&cobra.Command{
		Use:   "add <directory>",
		Short: "Upload the Markdown runbooks in a directory (re-running upserts)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			_, s, err := connected()
			if err != nil {
				return err
			}
			docs, err := client.CollectRunbooks(args[0])
			if err != nil {
				return err
			}
			report, err := client.New(s.Server).Ingest(cmd.Context(), docs)
			if err != nil {
				return err
			}
			enc := json.NewEncoder(cmd.OutOrStdout())
			enc.SetIndent("", "  ")
			return enc.Encode(report)
		},
	})
	return sources
}

func newAskCmd() *cobra.Command {
	var followup, asJSON bool
	c := &cobra.Command{
		Use:   "ask <question>",
		Short: "Ask the agent a question and print the verified, cited answer",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			path, s, err := connected()
			if err != nil {
				return err
			}
			thread := ""
			if followup {
				if s.ThreadID == "" {
					return errors.New("ask an initial question before continuing")
				}
				thread = s.ThreadID
			}
			ans, err := client.New(s.Server).Ask(cmd.Context(), args[0], thread)
			if err != nil {
				return err
			}
			s.ThreadID = ans.ThreadID
			if err := config.Save(path, s); err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			if asJSON {
				enc := json.NewEncoder(out)
				enc.SetIndent("", "  ")
				return enc.Encode(ans)
			}
			fmt.Fprintln(out, ans.Text)
			for i, src := range ans.Sources {
				label := src["doc_id"]
				if label == nil {
					label = src["document_title"]
				}
				fmt.Fprintf(out, "[%d] %v (%v)\n", i+1, src["section_heading"], label)
			}
			fmt.Fprintf(cmd.ErrOrStderr(), "\n%s · %.1f ms\n", ans.Outcome, ans.TotalMs)
			return nil
		},
	}
	c.Flags().BoolVar(&followup, "continue", false, "continue the previous conversation thread")
	c.Flags().BoolVar(&asJSON, "json", false, "print the answer, sources and timings as JSON")
	return c
}
