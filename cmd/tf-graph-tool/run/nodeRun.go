package run

import (
	"encoding/json"
	"fmt"

	"github.com/kubetrail/tfutil/cmd/tf-graph-tool/flags"
	"github.com/kubetrail/tfutil/pkg/proto/attr"
	"github.com/kubetrail/tfutil/pkg/tfutil"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	tf "github.com/wamuir/graft/tensorflow"
)

func NodeRun(cmd *cobra.Command, args []string) error {
	_ = viper.BindPFlag(flags.Name, cmd.Flag(flags.Name))
	_ = viper.BindPFlag(flags.InputFilename, cmd.Flag(flags.InputFilename))

	name := viper.GetString(flags.Name)
	inputFilename := viper.GetString(flags.InputFilename)

	if len(inputFilename) == 0 && len(args) == 1 {
		inputFilename = args[0]
	}

	graphDef, err := parseGraphDef(cmd, inputFilename)
	if err != nil {
		return err
	}

	graph, err := graphDef.Export("")
	if err != nil {
		return fmt.Errorf("failed to export graph: %w", err)
	}

	node, err := graphDef.GetNode(name)
	if err != nil {
		return fmt.Errorf("failed to get node %s: %w", name, err)
	}

	if node.NodeDef.Op != attr.Constant {
		return fmt.Errorf("node is not a constant and instead %s, please input names of const nodes only", node.NodeDef.Op)
	}

	// prepare data outputs from tensorflow run.
	// Identity is the final output point of the graph.
	fetches := []tf.Output{
		graph.Operation(name).Output(0),
	}

	// start new session
	sess, err := tf.NewSession(
		graph,
		&tf.SessionOptions{},
	)
	if err != nil {
		return fmt.Errorf("failed to create new tf session: %w", err)
	}

	defer func(sess *tf.Session) {
		err := sess.Close()
		if err != nil {
			panic(err)
		}
	}(sess)

	// run session feeding feeds and fetching fetches
	out, err := sess.Run(nil, fetches, nil)
	if err != nil {
		return fmt.Errorf("failed to run tf session: %w", err)
	}

	if len(out) != 1 {
		return fmt.Errorf("string reshape tf session output generated output length is not equal to 1")
	}

	tfTensor := out[0]
	switch tfTensor.DataType() {
	case tf.Bool:
		tensor := &tfutil.Tensor[bool]{}
		if err := tensor.Unmarshal(tfTensor); err != nil {
			return fmt.Errorf("failed to read tensor output from session run: %w", err)
		}
		jb, err := json.Marshal(tensor)
		if err != nil {
			return fmt.Errorf("failed to serialize tensor to json: %w", err)
		}
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	case tf.Uint8:
		tensor := &tfutil.Tensor[uint8]{}
		if err := tensor.Unmarshal(tfTensor); err != nil {
			return fmt.Errorf("failed to read tensor output from session run: %w", err)
		}
		jb, err := json.Marshal(tensor)
		if err != nil {
			return fmt.Errorf("failed to serialize tensor to json: %w", err)
		}
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	case tf.Uint16:
		tensor := &tfutil.Tensor[uint16]{}
		if err := tensor.Unmarshal(tfTensor); err != nil {
			return fmt.Errorf("failed to read tensor output from session run: %w", err)
		}
		jb, err := json.Marshal(tensor)
		if err != nil {
			return fmt.Errorf("failed to serialize tensor to json: %w", err)
		}
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	case tf.Uint32:
		tensor := &tfutil.Tensor[uint32]{}
		if err := tensor.Unmarshal(tfTensor); err != nil {
			return fmt.Errorf("failed to read tensor output from session run: %w", err)
		}
		jb, err := json.Marshal(tensor)
		if err != nil {
			return fmt.Errorf("failed to serialize tensor to json: %w", err)
		}
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	case tf.Uint64:
		tensor := &tfutil.Tensor[uint64]{}
		if err := tensor.Unmarshal(tfTensor); err != nil {
			return fmt.Errorf("failed to read tensor output from session run: %w", err)
		}
		jb, err := json.Marshal(tensor)
		if err != nil {
			return fmt.Errorf("failed to serialize tensor to json: %w", err)
		}
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	case tf.Int8:
		tensor := &tfutil.Tensor[int8]{}
		if err := tensor.Unmarshal(tfTensor); err != nil {
			return fmt.Errorf("failed to read tensor output from session run: %w", err)
		}
		jb, err := json.Marshal(tensor)
		if err != nil {
			return fmt.Errorf("failed to serialize tensor to json: %w", err)
		}
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	case tf.Int16:
		tensor := &tfutil.Tensor[int16]{}
		if err := tensor.Unmarshal(tfTensor); err != nil {
			return fmt.Errorf("failed to read tensor output from session run: %w", err)
		}
		jb, err := json.Marshal(tensor)
		if err != nil {
			return fmt.Errorf("failed to serialize tensor to json: %w", err)
		}
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	case tf.Int32:
		tensor := &tfutil.Tensor[int32]{}
		if err := tensor.Unmarshal(tfTensor); err != nil {
			return fmt.Errorf("failed to read tensor output from session run: %w", err)
		}
		jb, err := json.Marshal(tensor)
		if err != nil {
			return fmt.Errorf("failed to serialize tensor to json: %w", err)
		}
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	case tf.Int64:
		tensor := &tfutil.Tensor[int64]{}
		if err := tensor.Unmarshal(tfTensor); err != nil {
			return fmt.Errorf("failed to read tensor output from session run: %w", err)
		}
		jb, err := json.Marshal(tensor)
		if err != nil {
			return fmt.Errorf("failed to serialize tensor to json: %w", err)
		}
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	case tf.Double:
		tensor := &tfutil.Tensor[float64]{}
		if err := tensor.Unmarshal(tfTensor); err != nil {
			return fmt.Errorf("failed to read tensor output from session run: %w", err)
		}
		jb, err := json.Marshal(tensor)
		if err != nil {
			return fmt.Errorf("failed to serialize tensor to json: %w", err)
		}
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	case tf.Float:
		tensor := &tfutil.Tensor[float32]{}
		if err := tensor.Unmarshal(tfTensor); err != nil {
			return fmt.Errorf("failed to read tensor output from session run: %w", err)
		}
		jb, err := json.Marshal(tensor)
		if err != nil {
			return fmt.Errorf("failed to serialize tensor to json: %w", err)
		}
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), string(jb)); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
	default:
		return fmt.Errorf("tensor data type is not supported")
	}

	return nil
}
