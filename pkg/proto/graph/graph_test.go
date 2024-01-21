package graph

import (
	"bufio"
	"bytes"
	"encoding/json"
	"testing"

	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
)

func TestGraph(t *testing.T) {
	root := op.NewScope()

	x, err := tf.NewTensor([][]float64{{1, 2}, {3, 4}})
	if err != nil {
		t.Fatal(err)
	}

	X := op.Placeholder(
		root.SubScope("X"),
		x.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(2, 2),
		),
	)

	Y := op.Placeholder(
		root.SubScope("Y"),
		x.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(2, 2),
		),
	)

	_ = op.MatMul(root, X, Y)

	graph, err := root.Finalize()
	if err != nil {
		t.Fatal(err)
	}

	bb := &bytes.Buffer{}
	bw := bufio.NewWriter(bb)

	if _, err := graph.WriteTo(bw); err != nil {
		t.Fatal(err)
	}

	if err := bw.Flush(); err != nil {
		t.Fatal(err)
	}

	graphDef, err := NewGraphDef()
	if err != nil {
		t.Fatal(err)
	}

	if err := proto.Unmarshal(bb.Bytes(), graphDef); err != nil {
		t.Fatal(err)
	}

	pb, err := json.Marshal(graphDef)
	if err != nil {
		t.Fatal(err)
	}

	expectedPbTxt := `node:{name:"X/Placeholder"  op:"Placeholder"  attr:{key:"dtype"  value:{type:DT_DOUBLE}}  attr:{key:"shape"  value:{shape:{dim:{size:2}  dim:{size:2}}}}}  node:{name:"Y/Placeholder"  op:"Placeholder"  attr:{key:"dtype"  value:{type:DT_DOUBLE}}  attr:{key:"shape"  value:{shape:{dim:{size:2}  dim:{size:2}}}}}  node:{name:"MatMul"  op:"MatMul"  input:"X/Placeholder"  input:"Y/Placeholder"  attr:{key:"T"  value:{type:DT_DOUBLE}}  attr:{key:"transpose_a"  value:{b:false}}  attr:{key:"transpose_b"  value:{b:false}}}  versions:{producer:1645}  library:{}`
	graphDefExpected, err := NewGraphDef()
	if err != nil {
		t.Fatal(err)
	}

	if err := prototext.Unmarshal([]byte(expectedPbTxt), graphDefExpected); err != nil {
		t.Fatal(err)
	}

	pbExpected, err := json.Marshal(graphDefExpected)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(pb, pbExpected) {
		t.Fatal("graphdef does not match expected graphdef")
	}
}

func TestDef_UnmarshalJSON(t *testing.T) {
	graphData := `
{
  "node": [
    {
      "name": "testNode1",
      "op": "Const",
      "attr": {
        "dtype": {
          "Value": {
            "Type": 3
          }
        },
        "value": {
          "Value": {
            "Tensor": {
              "dtype": 3,
              "tensor_shape": {
                "dim": [
                  {
                    "size": 2
                  }
                ]
              },
              "tensor_content": "AQAAAOkDAAA="
            }
          }
        }
      }
    }
  ],
  "versions": {
    "producer": 1395
  },
  "library": {}
}
`

	graphDef, err := NewGraphDef()
	if err != nil {
		t.Fatal(err)
	}

	if err := json.Unmarshal([]byte(graphData), graphDef); err == nil {
		t.Fatal("expected json unmarshal to error out")
	}
}

func TestListNodesOptionWithInputs(t *testing.T) {
	root := op.NewScope()

	x, err := tf.NewTensor([][]float64{{1, 2}, {3, 4}})
	if err != nil {
		t.Fatal(err)
	}

	X := op.Placeholder(
		root.SubScope("X"),
		x.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(2, 2),
		),
	)

	Y := op.Placeholder(
		root.SubScope("Y"),
		x.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(2, 2),
		),
	)

	_ = op.MatMul(root, X, Y)

	graph, err := root.Finalize()
	if err != nil {
		t.Fatal(err)
	}

	graphDef, err := NewGraphDef()
	if err != nil {
		t.Fatal(err)
	}

	if err := graphDef.Import(graph); err != nil {
		t.Fatal(err)
	}

	opts := ListNodesOptionWithInputs("X/Placeholder")
	nodes := graphDef.ListNodes(opts)
	if len(nodes) != 1 || nodes[0] != "MatMul" {
		t.Fatal("listed not are not expected")
	}

	opts = ListNodesOptionWithInputs("X/Placeholder", "Y/Placeholder")
	nodes = graphDef.ListNodes(opts)
	if len(nodes) != 1 || nodes[0] != "MatMul" {
		t.Fatal("listed not are not expected")
	}
}
