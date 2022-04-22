/*
Copyright © 2022 kubetrail.io authors

*/
package cmd

import (
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/run"
	"github.com/spf13/cobra"
)

// graphMergeCmd represents the graphMerge command
var graphMergeCmd = &cobra.Command{
	Use:   "merge",
	Short: "merge input graphs",
	Long: `This commands merges input graphs assuming their node
names do not collide. Please use graph decode command and apply a prefix
to graph if there is a collision`,
	RunE: run.GraphMerge,
}

func init() {
	graphCmd.AddCommand(graphMergeCmd)
	f := graphMergeCmd.Flags()

	f.String(flags.OutputFormat, "json", "output graph format")
	f.StringSlice(flags.InputFilename, nil, "input graph filename")
	f.String(flags.OutputFilename, "-", "output filename (- for STDOUT)")
}
