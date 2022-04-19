package graph

import (
	"fmt"
	"testing"

	"github.com/kubetrail/tfutil/pkg/proto/attr"
	"github.com/kubetrail/tfutil/pkg/proto/node"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/attr_value_go_proto"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/types_go_proto"
	"google.golang.org/protobuf/proto"
)

func TestMatmul(t *testing.T) {
	x, err := tf.NewTensor([][]float64{{1, 2}, {3, 4}})
	if err != nil {
		t.Fatal(err)
	}

	y, err := tf.NewTensor([][]float64{{1, 2}, {3, 4}})
	if err != nil {
		t.Fatal(err)
	}

	attrX, err := attr.NewValueFromTensor(x)
	if err != nil {
		t.Fatal(err)
	}

	attrY, err := attr.NewValueFromTensor(y)
	if err != nil {
		t.Fatal(err)
	}

	nx, err := node.NewNodeDef("x", attr.Constant, attrX)
	if err != nil {
		t.Fatal(err)
	}

	ny, err := node.NewNodeDef("y", attr.Constant, attrY)
	if err != nil {
		t.Fatal(err)
	}

	attrValue := &attr.Value{
		Attr: map[string]*attr_value_go_proto.AttrValue{
			"T": {
				Value: &attr_value_go_proto.AttrValue_Type{
					Type: types_go_proto.DataType_DT_DOUBLE,
				},
			},
		},
	}

	mul, err := node.NewNodeDef("mul", "MatMul", attrValue, "x", "y")
	if err != nil {
		t.Fatal(err)
	}

	graphDef, err := NewGraphDef()
	if err != nil {
		t.Fatal(err)
	}

	graphDef.SetNodes(nx, ny, mul)

	pb, err := proto.Marshal(graphDef)
	if err != nil {
		t.Fatal(err)
	}

	graph := tf.NewGraph()
	if err := graph.Import(pb, ""); err != nil {
		t.Fatal(err)
	}

	fetches := []tf.Output{
		graph.Operation("mul").Output(0),
	}

	// start new session
	sess, err := tf.NewSession(
		graph,
		&tf.SessionOptions{},
	)
	if err != nil {
		t.Fatal(fmt.Errorf("failed to create new tf session: %w", err))
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
		t.Fatal(fmt.Errorf("failed to run tf session: %w", err))
	}

	if len(out) != 1 {
		t.Fatal("expected output len to be 1, got", len(out))
	}

	z, ok := (out[0].Value()).([][]float64)
	if !ok {
		t.Fatal("expected output to be [][]float64, failed in type assertion")
	}

	zExpected := [][]float64{{7, 10}, {15, 22}}
	if len(z) != len(zExpected) {
		t.Fatal("output [][]float64 length did not match expected")
	}

	for i := range z {
		if len(z[i]) != len(zExpected[i]) {
			t.Fatal("output [][]float64 at a row does not match expected length")
		}

		for j := range z[i] {
			if z[i][j] != zExpected[i][j] {
				t.Fatal("output value does not match expected output")
			}
		}
	}
}
