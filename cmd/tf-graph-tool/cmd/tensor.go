/*
Copyright © 2022 kubetrail.io authors

*/
package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// tensorCmd represents the tensor command
var tensorCmd = &cobra.Command{
	Use:   "tensor",
	Short: "tensor command group",
	Long:  `Encode tensors as constant nodes in graph`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("please use with subcommand")
	},
}

func init() {
	rootCmd.AddCommand(tensorCmd)
}
