package run

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/kubetrail/tfutil/pkg/proto/graph"
	"github.com/kubetrail/tfutil/pkg/tfutil"
	"github.com/spf13/cobra"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
)

type tensorDataType struct {
	GoDataType string `json:"goDataType,omitempty"`
}

func parseGraphDef(cmd *cobra.Command, inputFilename string) (*graph.Def, error) {
	var err error
	var b []byte
	var inputFormat string

	if inputFilename == "-" {
		b, err = io.ReadAll(cmd.InOrStdin())
		if err != nil {
			return nil, fmt.Errorf("failed to read stdin: %w", err)
		}
	} else {
		b, err = os.ReadFile(inputFilename)
		if err != nil {
			return nil, fmt.Errorf("failed to read input file: %w", err)
		}

		parts := strings.Split(inputFilename, ".")
		if len(parts) > 1 {
			ext := parts[len(parts)-1]
			switch v := strings.ToLower(ext); v {
			case "pb", "pbtxt", "pbtext", "json":
				inputFormat = v
			}
		}
	}

	graphDef, err := graph.NewGraphDef()
	if err != nil {
		return nil, fmt.Errorf("failed to get a new instance of graphdef: %w", err)
	}

	switch strings.ToLower(inputFormat) {
	case "pb":
		if err := proto.Unmarshal(b, graphDef); err != nil {
			return nil, fmt.Errorf("failed to unmarshal input: %w", err)
		}
	case "pbtxt", "pbtext":
		if err := prototext.Unmarshal(b, graphDef); err != nil {
			return nil, fmt.Errorf("failed to unmarshal as proto text: %w", err)
		}
	case "json":
		if err := json.Unmarshal(b, graphDef); err != nil {
			return nil, fmt.Errorf("failed to decode as json text: %w", err)
		}
	case "":
		var terr error
		if err := proto.Unmarshal(b, graphDef); err != nil {
			terr = fmt.Errorf("failed to unmarshal input as binary proto format: %w", err)
		} else {
			break
		}

		if err := prototext.Unmarshal(b, graphDef); err != nil {
			terr = fmt.Errorf("failed to unmarshal input as proto text: %w, %s", terr, err)
		} else {
			break
		}

		if err := json.Unmarshal(b, graphDef); err != nil {
			terr = fmt.Errorf("failed to unmarshal input as json: %w, %s", terr, err)
		} else {
			break
		}

		if terr != nil {
			return nil, terr
		}
	}

	return graphDef, nil
}

func parseTensor(cmd *cobra.Command, inputFilename string) (*tf.Tensor, error) {
	var err error
	var b []byte

	if inputFilename == "-" {
		b, err = io.ReadAll(cmd.InOrStdin())
		if err != nil {
			return nil, fmt.Errorf("failed to read stdin: %w", err)
		}
	} else {
		b, err = os.ReadFile(inputFilename)
		if err != nil {
			return nil, fmt.Errorf("failed to read input file: %w", err)
		}
	}

	dt := &tensorDataType{}
	if err := json.Unmarshal(b, dt); err != nil {
		return nil, fmt.Errorf("failed to get tensor data type: %w", err)
	}

	var tfTensor *tf.Tensor

	switch dt.GoDataType {
	case "bool":
		tensor := &tfutil.Tensor[bool]{}
		if err := json.Unmarshal(b, tensor); err != nil {
			return nil, fmt.Errorf("failed to parse tensor data: %w", err)
		}

		tfTensor, err = tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to encode tf tensor: %w", err)
		}
	case "uint8", "byte":
		tensor := &tfutil.Tensor[uint8]{}
		if err := json.Unmarshal(b, tensor); err != nil {
			return nil, fmt.Errorf("failed to parse tensor data: %w", err)
		}

		tfTensor, err = tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to encode tf tensor: %w", err)
		}
	case "uint16":
		tensor := &tfutil.Tensor[uint16]{}
		if err := json.Unmarshal(b, tensor); err != nil {
			return nil, fmt.Errorf("failed to parse tensor data: %w", err)
		}

		tfTensor, err = tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to encode tf tensor: %w", err)
		}
	case "uint32":
		tensor := &tfutil.Tensor[uint32]{}
		if err := json.Unmarshal(b, tensor); err != nil {
			return nil, fmt.Errorf("failed to parse tensor data: %w", err)
		}

		tfTensor, err = tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to encode tf tensor: %w", err)
		}
	case "uint64":
		tensor := &tfutil.Tensor[uint64]{}
		if err := json.Unmarshal(b, tensor); err != nil {
			return nil, fmt.Errorf("failed to parse tensor data: %w", err)
		}

		tfTensor, err = tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to encode tf tensor: %w", err)
		}
	case "int8":
		tensor := &tfutil.Tensor[int8]{}
		if err := json.Unmarshal(b, tensor); err != nil {
			return nil, fmt.Errorf("failed to parse tensor data: %w", err)
		}

		tfTensor, err = tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to encode tf tensor: %w", err)
		}
	case "int16":
		tensor := &tfutil.Tensor[int16]{}
		if err := json.Unmarshal(b, tensor); err != nil {
			return nil, fmt.Errorf("failed to parse tensor data: %w", err)
		}

		tfTensor, err = tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to encode tf tensor: %w", err)
		}
	case "int32":
		tensor := &tfutil.Tensor[int32]{}
		if err := json.Unmarshal(b, tensor); err != nil {
			return nil, fmt.Errorf("failed to parse tensor data: %w", err)
		}

		tfTensor, err = tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to encode tf tensor: %w", err)
		}
	case "int64":
		tensor := &tfutil.Tensor[int64]{}
		if err := json.Unmarshal(b, tensor); err != nil {
			return nil, fmt.Errorf("failed to parse tensor data: %w", err)
		}

		tfTensor, err = tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to encode tf tensor: %w", err)
		}
	case "float32":
		tensor := &tfutil.Tensor[float32]{}
		if err := json.Unmarshal(b, tensor); err != nil {
			return nil, fmt.Errorf("failed to parse tensor data: %w", err)
		}

		tfTensor, err = tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to encode tf tensor: %w", err)
		}
	case "float64":
		tensor := &tfutil.Tensor[float64]{}
		if err := json.Unmarshal(b, tensor); err != nil {
			return nil, fmt.Errorf("failed to parse tensor data: %w", err)
		}

		tfTensor, err = tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to encode tf tensor: %w", err)
		}
	default:
		return nil, fmt.Errorf("data type %s is not supported", dt.GoDataType)
	}

	return tfTensor, nil
}

func printGraphDef(cmd *cobra.Command, graphDef *graph.Def, outputFilename, outputFormat string) error {
	var b []byte
	var err error
	switch strings.ToLower(outputFormat) {
	case "pb":
		b, err = proto.Marshal(graphDef)
		if err != nil {
			return fmt.Errorf("failed to marshal output as proto binary: %w", err)
		}
	case "pbtxt", "pbtext":
		b, err = prototext.Marshal(graphDef)
		if err != nil {
			return fmt.Errorf("failed to marshal output as proto text: %w", err)
		}
	case "json":
		b, err = json.MarshalIndent(graphDef, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal output as json: %w", err)
		}
	case "dot":
		b, err = graphDef.PrintDotNotation()
		if err != nil {
			return fmt.Errorf("failed to print graph def as dot notation: %w", err)
		}
	default:
		return fmt.Errorf("invalid output format %s, pl. specify pb, pbtxt or json", outputFormat)
	}

	if outputFilename == "-" {
		if _, err := cmd.OutOrStdout().Write(b); err != nil {
			return fmt.Errorf("failed to write to output: %w", err)
		}
		return nil
	}

	if err := os.WriteFile(outputFilename, b, 0644); err != nil {
		return fmt.Errorf("failed to write output file: %w", err)
	}

	return nil
}
