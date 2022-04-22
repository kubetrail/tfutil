package run

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/pkg/proto/attr"
	"github.com/kubetrail/tfutil/pkg/proto/graph"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func NodeList(cmd *cobra.Command, args []string) error {
	_ = viper.BindPFlag(flags.Selector, cmd.Flag(flags.Selector))
	_ = viper.BindPFlag(flags.InputFilename, cmd.Flag(flags.InputFilename))

	selectors := viper.GetStringSlice(flags.Selector)
	inputFilename := viper.GetString(flags.InputFilename)

	if len(inputFilename) == 0 && len(args) == 1 {
		inputFilename = args[0]
	}

	graphDef, err := parseGraphDef(cmd, inputFilename)
	if err != nil {
		return err
	}

	opts := make([]graph.ListNodesOption, 0, len(selectors))
	inputs := make([]string, 0, len(selectors))

	for _, selector := range selectors {
		parts := strings.Split(selector, "=")
		if len(parts) != 2 {
			return fmt.Errorf("selector %s not formatted in key=value format", selector)
		}

		key, value := parts[0], parts[1]

		switch strings.ToLower(value) {
		case "const", "constant":
			value = attr.Constant
		case "placeholder":
			value = attr.Placeholder
		}

		switch strings.ToLower(key) {
		case "op", "operation":
			opts = append(opts, graph.ListNodesOptionOp(value))
		case "input":
			inputs = append(inputs, value)
		default:
			return fmt.Errorf("selector key %s is not valid", key)
		}

		if len(inputs) > 0 {
			opts = append(opts, graph.ListNodesOptionWithInputs(inputs...))
		}
	}

	nodes := graphDef.ListNodes(opts...)
	jb, err := json.MarshalIndent(nodes, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to json serialize node list: %w", err)
	}

	if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
		return fmt.Errorf("failed to write to output: %w", err)
	}

	return nil
}
