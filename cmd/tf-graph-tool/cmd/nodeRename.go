/*
Copyright © 2022 kubetrail.io authors

*/
package cmd

import (
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/run"
	"github.com/spf13/cobra"
)

// nodeRenameCmd represents the nodeRename command
var nodeRenameCmd = &cobra.Command{
	Use:   "rename",
	Short: "rename the node",
	Long:  `This command renames a node and produces a new graph`,
	RunE:  run.NodeRename,
}

func init() {
	nodeCmd.AddCommand(nodeRenameCmd)
	f := nodeRenameCmd.Flags()

	f.String(flags.OutputFormat, "json", "output graph format")
	f.String(flags.InputFilename, "", "input filename (- for STDIN)")
	f.String(flags.OutputFilename, "-", "output filename (- for STDOUT)")
	f.String(flags.Name, "", "node name")
	f.String(flags.NewName, "", "new name for the node")
}
