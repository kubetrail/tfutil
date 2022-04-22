package run

import (
	"fmt"

	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func NodeRename(cmd *cobra.Command, args []string) error {
	_ = viper.BindPFlag(flags.Name, cmd.Flag(flags.Name))
	_ = viper.BindPFlag(flags.NewName, cmd.Flag(flags.NewName))
	_ = viper.BindPFlag(flags.InputFilename, cmd.Flag(flags.InputFilename))
	_ = viper.BindPFlag(flags.OutputFilename, cmd.Flag(flags.OutputFilename))
	_ = viper.BindPFlag(flags.OutputFormat, cmd.Flag(flags.OutputFormat))

	name := viper.GetString(flags.Name)
	newName := viper.GetString(flags.NewName)
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

	if err := graphDef.RenameNode(name, newName); err != nil {
		return fmt.Errorf("failed to rename node: %w", err)
	}

	return printGraphDef(cmd, graphDef, outputFilename, outputFormat)
}