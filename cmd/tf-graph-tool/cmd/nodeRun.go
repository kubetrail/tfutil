/*
Copyright © 2022 kubetrail.io authors

*/
package cmd

import (
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/run"
	"github.com/spf13/cobra"
)

// nodeRunCmd represents the nodeRun command
var nodeRunCmd = &cobra.Command{
	Use:   "run",
	Short: "get raw tensor output from a node",
	Long: `This command is mainly intended for extracting constant node
tensor values or values from nodes that can execute without providing
any inputs to the graph.`,
	RunE: run.NodeRun,
}

func init() {
	nodeCmd.AddCommand(nodeRunCmd)
	f := nodeRunCmd.Flags()

	f.String(flags.InputFilename, "", "input graph filename")
	f.String(flags.Name, "Identity", "node name")
}
