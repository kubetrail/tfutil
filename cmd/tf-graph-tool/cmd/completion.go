/*
Copyright © 2022 kubetrail.io authors
*/
package cmd

import (
	"os"
	"strings"

	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/run"
	"github.com/spf13/cobra"
)

var longCompletionCmd = `To load completions:

Bash:

  $ source <(appName completion bash)

  # To load completions for each session, execute once:
  # Linux:
  $ appName completion bash > /etc/bash_completion.d/appName
  # macOS:
  $ appName completion bash > /usr/local/etc/bash_completion.d/appName

Zsh:

  # If shell completion is not already enabled in your environment,
  # you will need to enable it.  You can execute the following once:

  $ echo "autoload -U compinit; compinit" >> ~/.zshrc

  # To load completions for each session, execute once:
  $ appName completion zsh > "${fpath[1]}/_appName"

  # You will need to start a new shell for this setup to take effect.

fish:

  $ appName completion fish | source

  # To load completions for each session, execute once:
  $ appName completion fish > ~/.config/fish/completions/appName.fish

PowerShell:

  PS> appName completion powershell | Out-String | Invoke-Expression

  # To load completions for every new session, run:
  PS> appName completion powershell > appName.ps1
  # and source this file from your PowerShell profile.
`

// completionCmd represents the completion command
var completionCmd = &cobra.Command{
	Use:   "completion",
	Short: "Generate shell completion",
	Long: strings.ReplaceAll(
		longCompletionCmd,
		"appName",
		run.AppName),
	DisableFlagsInUseLine: true,
	ValidArgs:             []string{"bash", "zsh", "fish", "powershell"},
	Args:                  cobra.ExactValidArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		switch args[0] {
		case "bash":
			_ = cmd.Root().GenBashCompletion(os.Stdout)
		case "zsh":
			_ = cmd.Root().GenZshCompletion(os.Stdout)
		case "fish":
			_ = cmd.Root().GenFishCompletion(os.Stdout, true)
		case "powershell":
			_ = cmd.Root().GenPowerShellCompletionWithDesc(os.Stdout)
		}
	},
}

func init() {
	rootCmd.AddCommand(completionCmd)
}
