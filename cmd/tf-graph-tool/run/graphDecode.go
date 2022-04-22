package run

import (
	"fmt"

	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func GraphDecode(cmd *cobra.Command, args []string) error {
	_ = viper.BindPFlag(flags.OutputFormat, cmd.Flag(flags.OutputFormat))
	_ = viper.BindPFlag(flags.InputFilename, cmd.Flag(flags.InputFilename))
	_ = viper.BindPFlag(flags.OutputFilename, cmd.Flag(flags.OutputFilename))
	_ = viper.BindPFlag(flags.AddPrefix, cmd.Flag(flags.AddPrefix))

	inputFilename := viper.GetString(flags.InputFilename)
	outputFilename := viper.GetString(flags.OutputFilename)
	outputFormat := viper.GetString(flags.OutputFormat)
	prefix := viper.GetString(flags.AddPrefix)

	if len(inputFilename) == 0 && len(args) == 1 {
		inputFilename = args[0]
	}

	graphDef, err := parseGraphDef(cmd, inputFilename)
	if err != nil {
		return err
	}

	if len(prefix) > 0 {
		if err := graphDef.ApplyPrefix(prefix); err != nil {
			return fmt.Errorf("failed to add prefix to graphdef: %w", err)
		}
	}

	return printGraphDef(cmd, graphDef, outputFilename, outputFormat)
}
