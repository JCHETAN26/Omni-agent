// Command omni is the OmniAgent developer CLI.
package main

import (
	"os"

	"github.com/JCHETAN26/Omni-agent/cli/internal/cmd"
)

func main() { os.Exit(cmd.Execute()) }
