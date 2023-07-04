/*
Copyright © 2022 kubetrail.io authors
*/
package cmd

import (
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/run"
	"github.com/spf13/cobra"
)

// nodeDeleteCmd represents the nodeDelete command
var nodeDeleteCmd = &cobra.Command{
	Use:   "delete",
	Short: "delete nodes from graph",
	Long:  `Delete nodes from graph`,
	RunE:  run.NodeDelete,
}

func init() {
	nodeCmd.AddCommand(nodeDeleteCmd)
	f := nodeDeleteCmd.Flags()

	f.String(flags.OutputFormat, "json", "output graph format")
	f.String(flags.InputFilename, "", "input filename (- for STDIN)")
	f.String(flags.OutputFilename, "-", "output filename (- for STDOUT)")
	f.StringSlice(flags.Name, nil, "node name to delete")
}
