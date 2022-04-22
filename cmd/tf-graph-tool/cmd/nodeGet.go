/*
Copyright © 2022 kubetrail.io authors

*/
package cmd

import (
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/run"
	"github.com/spf13/cobra"
)

// nodeGetCmd represents the nodeGet command
var nodeGetCmd = &cobra.Command{
	Use:   "get",
	Short: "get node details",
	Long:  `Retrieve details on a particular node`,
	RunE:  run.NodeGet,
}

func init() {
	nodeCmd.AddCommand(nodeGetCmd)
	f := nodeGetCmd.Flags()

	f.String(flags.OutputFormat, "json", "output graph format")
	f.String(flags.InputFilename, "", "input filename (- for STDIN)")
	f.String(flags.OutputFilename, "-", "output filename (- for STDOUT)")
	f.String(flags.Name, "", "node name")
}
