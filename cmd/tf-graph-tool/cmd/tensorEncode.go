/*
Copyright © 2022 kubetrail.io authors

*/
package cmd

import (
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/run"
	"github.com/spf13/cobra"
)

// tensorEncodeCmd represents the tensorEncode command
var tensorEncodeCmd = &cobra.Command{
	Use:   "encode",
	Short: "encode tensor as const node",
	Long:  `Encode tensor as constant node in a graph`,
	RunE:  run.TensorEncode,
}

func init() {
	tensorCmd.AddCommand(tensorEncodeCmd)
	f := tensorEncodeCmd.Flags()

	f.String(flags.OutputFormat, "json", "output graph format")
	f.String(flags.InputFilename, "", "input filename containing tensor (- for STDIN)")
	f.String(flags.OutputFilename, "-", "output graph filename (- for STDOUT)")
	f.String(flags.Name, "", "node name")
}
