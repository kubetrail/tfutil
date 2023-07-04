/*
Copyright © 2022 kubetrail.io authors
*/
package cmd

import (
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/run"
	"github.com/spf13/cobra"
)

// graphDecodeCmd represents the graphDecode command
var graphDecodeCmd = &cobra.Command{
	Use:   "decode",
	Short: "decode graph from one format to another",
	Long: `Decode graph into a human readable format such as proto text
or json.`,
	RunE: run.GraphDecode,
}

func init() {
	graphCmd.AddCommand(graphDecodeCmd)
	f := graphDecodeCmd.Flags()

	f.String(flags.OutputFormat, "json", "output graph format")
	f.String(flags.InputFilename, "", "input filename (- for STDIN)")
	f.String(flags.OutputFilename, "-", "output filename (- for STDOUT)")
	f.String(flags.AddPrefix, "", "add prefix to output graph node names")
}
