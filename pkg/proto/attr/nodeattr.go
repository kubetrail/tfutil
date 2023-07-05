package attr

import (
	"bufio"
	"bytes"
	"fmt"

	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/core/framework/attr_value_go_proto"
	"github.com/wamuir/graft/tensorflow/core/framework/tensor_go_proto"
	"github.com/wamuir/graft/tensorflow/core/framework/tensor_shape_go_proto"
	"github.com/wamuir/graft/tensorflow/core/framework/types_go_proto"
)

// NewValueFromTensor embeds an input tensor into attr values that become part of
// the node in the graph. In particular, a tensor is represented as an
// attribute via its serialized form along with its shape and data type.
// Furthermore, a separate key just representing data type is added in order
// for this to work in tensorflow session runs
func NewValueFromTensor(tensor *tf.Tensor) (*Value, error) {
	// dimension and shape calculation
	dim := make([]*tensor_shape_go_proto.TensorShapeProto_Dim, len(tensor.Shape()))
	for i, v := range tensor.Shape() {
		dim[i] = &tensor_shape_go_proto.TensorShapeProto_Dim{
			Size: v,
			Name: "",
		}
	}
	tensorShape := &tensor_shape_go_proto.TensorShapeProto{
		Dim:         dim,
		UnknownRank: false,
	}

	// data type calculation
	dType := types_go_proto.DataType(tensor.DataType())

	// tensor content calculation
	bb := &bytes.Buffer{}
	bw := bufio.NewWriter(bb)
	if _, err := tensor.WriteContentsTo(bw); err != nil {
		return nil, fmt.Errorf("failed to serialize tensor contents: %w", err)
	}
	if err := bw.Flush(); err != nil {
		return nil, fmt.Errorf("failed to flush buffer: %w", err)
	}

	dtype := &attr_value_go_proto.AttrValue{
		Value: &attr_value_go_proto.AttrValue_Type{Type: dType},
	}

	value := &attr_value_go_proto.AttrValue{
		Value: &attr_value_go_proto.AttrValue_Tensor{
			Tensor: &tensor_go_proto.TensorProto{
				Dtype:             dType,
				TensorShape:       tensorShape,
				VersionNumber:     0,
				TensorContent:     bb.Bytes(),
				HalfVal:           nil,
				FloatVal:          nil,
				DoubleVal:         nil,
				IntVal:            nil,
				StringVal:         nil,
				ScomplexVal:       nil,
				Int64Val:          nil,
				BoolVal:           nil,
				DcomplexVal:       nil,
				ResourceHandleVal: nil,
				VariantVal:        nil,
				Uint32Val:         nil,
				Uint64Val:         nil,
			},
		},
	}

	return &Value{
		Attr: map[string]*attr_value_go_proto.AttrValue{
			"value": value,
			"dtype": dtype,
		},
	}, nil
}

func NewValueFromDataType(dataType tf.DataType) *Value {
	return &Value{
		Attr: map[string]*attr_value_go_proto.AttrValue{
			"dtype": {
				Value: &attr_value_go_proto.AttrValue_Type{
					Type: types_go_proto.DataType(dataType),
				},
			},
		},
	}
}
