package node

import (
	"bufio"
	"bytes"

	"github.com/kubetrail/tfutil/pkg/proto/attr"
	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/core/framework/graph_go_proto"
	"github.com/wamuir/graft/tensorflow/core/framework/node_def_go_proto"
	"github.com/wamuir/graft/tensorflow/op"
	"google.golang.org/protobuf/proto"
)

// NewConstantNode creates a new const node based on a tensor
func NewConstantNode(name string, tensor *tf.Tensor) (*Def, error) {
	attrValue, err := attr.NewValueFromTensor(tensor)
	if err != nil {
		return nil, err
	}
	nodeDef := &node_def_go_proto.NodeDef{
		Name:                  name,
		Op:                    attr.Constant,
		Input:                 nil,
		Device:                "",
		Attr:                  attrValue.Attr,
		ExperimentalDebugInfo: nil,
		ExperimentalType:      nil,
	}
	return &Def{NodeDef: nodeDef}, nil
}

// NewPlaceholderNode creates a new placeholder node for a given
// data type
func NewPlaceholderNode(name string, dType tf.DataType) *Def {
	attrValue := attr.NewValueFromDataType(dType)
	nodeDef := &node_def_go_proto.NodeDef{
		Name:                  name,
		Op:                    attr.Placeholder,
		Input:                 nil,
		Device:                "",
		Attr:                  attrValue.Attr,
		ExperimentalDebugInfo: nil,
		ExperimentalType:      nil,
	}
	return &Def{NodeDef: nodeDef}
}

// NewNodesFromTfOp creates a graph using input operators
// and then extracts nodes from it. Please note that the operator
// signature only requires op.Scope as input, so it may be necessary
// to input closures. For instance, op.MatMul requires two tf.Output
// inputs
func NewNodesFromTfOp(operators ...func(scope *op.Scope)) ([]*Def, error) {
	root := op.NewScope()
	for _, operator := range operators {
		operator(root)
	}
	graph, err := root.Finalize()
	if err != nil {
		return nil, err
	}
	bb := &bytes.Buffer{}
	bw := bufio.NewWriter(bb)

	if _, err := graph.WriteTo(bw); err != nil {
		return nil, err
	}

	if err := bw.Flush(); err != nil {
		return nil, err
	}

	graphDef := &graph_go_proto.GraphDef{}
	if err := proto.Unmarshal(bb.Bytes(), graphDef); err != nil {
		return nil, err
	}

	nodes := make([]*Def, len(graphDef.Node))
	for i, node := range graphDef.Node {
		nodes[i] = &Def{NodeDef: node}
	}

	return nodes, nil
}
