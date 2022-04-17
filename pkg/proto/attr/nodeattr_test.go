package attr

import (
	"bytes"
	"encoding/json"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/graph_go_proto"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/node_def_go_proto"
	"google.golang.org/protobuf/encoding/prototext"
)

func TestGetNodeAttrFromTensor(t *testing.T) {
	value := make([][]int32, 3)
	for i := range value {
		value[i] = make([]int32, 4)
		for j := range value[i] {
			value[i][j] = int32(i*4 + j)
		}
	}

	tensor, err := tf.NewTensor(value)
	if err != nil {
		t.Fatal(err)
	}

	attr, err := NewValueFromTensor(tensor)
	if err != nil {
		t.Fatal(err)
	}

	graphDef := &graph_go_proto.GraphDef{
		Node: []*node_def_go_proto.NodeDef{
			{
				Name:                  "weights",
				Op:                    "Const",
				Input:                 nil,
				Device:                "",
				Attr:                  attr.Attr,
				ExperimentalDebugInfo: nil,
				ExperimentalType:      nil,
			},
		},
	}

	expected := "node:{name:\"weights\"  op:\"Const\"  attr:{key:\"dtype\"  value:{type:DT_INT32}}  attr:{key:\"value\"  value:{tensor:{dtype:DT_INT32  tensor_shape:{dim:{size:3}  dim:{size:4}}  tensor_content:\"\\x00\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x02\\x00\\x00\\x00\\x03\\x00\\x00\\x00\\x04\\x00\\x00\\x00\\x05\\x00\\x00\\x00\\x06\\x00\\x00\\x00\\x07\\x00\\x00\\x00\\x08\\x00\\x00\\x00\\t\\x00\\x00\\x00\\n\\x00\\x00\\x00\\x0b\\x00\\x00\\x00\"}}}}"

	graphDefExpected := &graph_go_proto.GraphDef{}
	if err := prototext.Unmarshal([]byte(expected), graphDefExpected); err != nil {
		t.Fatal(err)
	}

	pb, err := json.Marshal(graphDef)
	if err != nil {
		t.Fatal(err)
	}

	pbExpected, err := json.Marshal(graphDefExpected)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(pb, pbExpected) {
		t.Fatal("marshaled proto output is not the same as expected")
	}
}
