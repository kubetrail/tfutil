package run

import (
	"fmt"

	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/pkg/proto/graph"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func NodeGet(cmd *cobra.Command, args []string) error {
	_ = viper.BindPFlag(flags.Name, cmd.Flag(flags.Name))
	_ = viper.BindPFlag(flags.InputFilename, cmd.Flag(flags.InputFilename))
	_ = viper.BindPFlag(flags.OutputFilename, cmd.Flag(flags.OutputFilename))
	_ = viper.BindPFlag(flags.OutputFormat, cmd.Flag(flags.OutputFormat))

	name := viper.GetString(flags.Name)
	inputFilename := viper.GetString(flags.InputFilename)
	outputFilename := viper.GetString(flags.OutputFilename)
	outputFormat := viper.GetString(flags.OutputFormat)

	if len(inputFilename) == 0 && len(args) == 1 {
		inputFilename = args[0]
	}

	graphDef, err := parseGraphDef(cmd, inputFilename)
	if err != nil {
		return err
	}

	node, err := graphDef.GetNode(name)
	if err != nil {
		return fmt.Errorf("failed to get node %s: %w", name, err)
	}

	graphDef, err = graph.NewGraphDef()
	if err != nil {
		return fmt.Errorf("failed to create a new graph: %w", err)
	}

	graphDef.SetNodes(node)

	return printGraphDef(cmd, graphDef, outputFilename, outputFormat)
}
