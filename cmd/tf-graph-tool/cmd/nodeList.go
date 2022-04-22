/*
Copyright © 2022 kubetrail.io authors

*/
package cmd

import (
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/run"
	"github.com/spf13/cobra"
)

// nodeListCmd represents the nodeList command
var nodeListCmd = &cobra.Command{
	Use:   "list",
	Short: "list nodes",
	Long:  `List nodes in the graph`,
	RunE:  run.NodeList,
}

func init() {
	nodeCmd.AddCommand(nodeListCmd)
	f := nodeListCmd.Flags()

	f.StringSlice(flags.Selector, nil, "selector in format key=value")
	f.String(flags.InputFilename, "", "input graph filename")
}
