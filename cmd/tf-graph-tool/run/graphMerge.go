package run

import (
	"fmt"

	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/pkg/proto/graph"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func GraphMerge(cmd *cobra.Command, args []string) error {
	_ = viper.BindPFlag(flags.OutputFormat, cmd.Flag(flags.OutputFormat))
	_ = viper.BindPFlag(flags.InputFilename, cmd.Flag(flags.InputFilename))
	_ = viper.BindPFlag(flags.OutputFilename, cmd.Flag(flags.OutputFilename))

	inputFilenames := viper.GetStringSlice(flags.InputFilename)
	outputFilename := viper.GetString(flags.OutputFilename)
	outputFormat := viper.GetString(flags.OutputFormat)

	for _, arg := range args {
		inputFilenames = append(inputFilenames, arg)
	}

	graphDef, err := graph.NewGraphDef()
	if err != nil {
		return fmt.Errorf("failed to create a new instance of graph: %w", err)
	}

	for _, inputFilename := range inputFilenames {
		def, err := parseGraphDef(cmd, inputFilename)
		if err != nil {
			return err
		}

		if err := graphDef.Append(def); err != nil {
			return fmt.Errorf("failed to append graph %s: %w", inputFilename, err)
		}
	}

	return printGraphDef(cmd, graphDef, outputFilename, outputFormat)
}
