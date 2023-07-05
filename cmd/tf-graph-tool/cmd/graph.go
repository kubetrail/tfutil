/*
Copyright © 2022 kubetrail.io authors
*/
package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// graphCmd represents the graph command
var graphCmd = &cobra.Command{
	Use:   "graph",
	Short: "graph encode decode command group",
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("please use with a subcommand")
	},
}

func init() {
	rootCmd.AddCommand(graphCmd)
}
