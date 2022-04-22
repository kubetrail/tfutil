/*
Copyright © 2022 kubetrail.io authors

*/
package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// nodeCmd represents the node command
var nodeCmd = &cobra.Command{
	Use:   "node",
	Short: "node related command category",
	Long: `Perform node related operations such as
node listing based on name or operation selector,
node renaming etc.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("please use a subcommand for this command")
	},
}

func init() {
	rootCmd.AddCommand(nodeCmd)
}
