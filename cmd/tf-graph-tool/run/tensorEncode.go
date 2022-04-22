package run

import (
	"fmt"

	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/pkg/proto/graph"
	"github.com/kubetrail/tfutil/pkg/proto/node"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func TensorEncode(cmd *cobra.Command, args []string) error {
	_ = viper.BindPFlag(flags.Name, cmd.Flag(flags.Name))
	_ = viper.BindPFlag(flags.OutputFormat, cmd.Flag(flags.OutputFormat))
	_ = viper.BindPFlag(flags.InputFilename, cmd.Flag(flags.InputFilename))
	_ = viper.BindPFlag(flags.OutputFilename, cmd.Flag(flags.OutputFilename))

	name := viper.GetString(flags.Name)
	inputFilename := viper.GetString(flags.InputFilename)
	outputFilename := viper.GetString(flags.OutputFilename)
	outputFormat := viper.GetString(flags.OutputFormat)

	if len(inputFilename) == 0 && len(args) == 1 {
		inputFilename = args[0]
	}

	tensor, err := parseTensor(cmd, inputFilename)
	if err != nil {
		return fmt.Errorf("failed to parse tensor: %w", err)
	}

	tensorNode, err := node.NewConstantNode(name, tensor)
	if err != nil {
		return fmt.Errorf("failed to generate const node from tensor: %w", err)
	}

	graphDef, err := graph.NewGraphDef()
	if err != nil {
		return err
	}

	graphDef.SetNodes(tensorNode)

	return printGraphDef(cmd, graphDef, outputFilename, outputFormat)
}
