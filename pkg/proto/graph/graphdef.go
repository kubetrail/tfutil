package graph

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/kubetrail/tfutil/pkg/proto/attr"
	"github.com/kubetrail/tfutil/pkg/proto/node"
	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/core/framework/graph_go_proto"
	"github.com/wamuir/graft/tensorflow/core/framework/node_def_go_proto"
	"github.com/wamuir/graft/tensorflow/op"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// make sure following interfaces are satisfied
var (
	_ proto.Message    = (*Def)(nil) // for protobuf
	_ json.Marshaler   = (*Def)(nil) // for json
	_ json.Unmarshaler = (*Def)(nil) // for json
	_ fmt.Stringer     = (*Def)(nil) // for printing
)

// Def wraps proto definition of graphdef to define a few
// utility methods on it
type Def struct {
	graphDef *graph_go_proto.GraphDef
}

// ListNodesOption is a func that answers yes/no on a node.
// Such options can be provided to filter node list
type ListNodesOption func(def *node.Def) bool

// ProtoReflect defines how receiver can be proto marshaled and unmarshaled
func (g *Def) ProtoReflect() protoreflect.Message {
	return g.graphDef.ProtoReflect()
}

// MarshalJSON defines how receiver can be json marshaled
func (g *Def) MarshalJSON() ([]byte, error) {
	return json.Marshal(g.graphDef)
}

// UnmarshalJSON defines how receiver can be json unmarshaled
func (g *Def) UnmarshalJSON(data []byte) error {
	return fmt.Errorf("json unmarshal is currently not supported")
}

// String defines how receiver can be printed
func (g *Def) String() string {
	pb, err := prototext.Marshal(g)
	if err != nil {
		panic(err)
	}
	return string(pb)
}

// PrintDotNotation defines how receiver can be printed as a dot notation
// for visualization via graphviz
// https://graphviz.org/doc/info/lang.html
func (g *Def) PrintDotNotation() ([]byte, error) {
	bb := &bytes.Buffer{}
	bw := bufio.NewWriter(bb)

	colorMap := map[string]string{
		attr.Constant:    "aquamarine3",
		attr.Placeholder: "antiquewhite3",
	}

	for _, nodeDef := range g.graphDef.Node {
		color, ok := colorMap[nodeDef.Op]
		if ok {
			if _, err := fmt.Fprintf(bw, "\"%s\" [fillcolor=\"%s\", style=\"filled\", label=\"%s\"]\n", nodeDef.Name, color, nodeDef.Name); err != nil {
				return nil, fmt.Errorf("failed to write to buffer: %w", err)
			}
		} else {
			if _, err := fmt.Fprintf(bw, "\"%s\" [label=\"%s\"]\n", nodeDef.Name, nodeDef.Name); err != nil {
				return nil, fmt.Errorf("failed to write to buffer: %w", err)
			}
		}
		for _, input := range nodeDef.Input {
			if _, err := fmt.Fprintf(bw, "\"%s\" -> \"%s\"\n", input, nodeDef.Name); err != nil {
				return nil, fmt.Errorf("failed to write to buffer: %w", err)
			}
		}
	}

	if err := bw.Flush(); err != nil {
		return nil, fmt.Errorf("failed to flush buffer: %w", err)
	}

	return bb.Bytes(), nil
}

// NewGraphDef provides a new instance of Def
func NewGraphDef() (*Def, error) {
	graphDef, err := newGraphDef()
	if err != nil {
		return nil, err
	}
	return &Def{graphDef: graphDef}, nil
}

// LoadFile imports graph from a file formatted as either
// binary protobuf, protobuf text or json
func LoadFile(filename string) (*Def, error) {
	pb, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	graphDef := &Def{graphDef: &graph_go_proto.GraphDef{}}
	var tErr error
	if err := proto.Unmarshal(pb, graphDef); err != nil {
		tErr = fmt.Errorf("failed to decode graph as proto binary: %w", err)
	} else {
		return graphDef, nil
	}

	if err := prototext.Unmarshal(pb, graphDef); err != nil {
		tErr = fmt.Errorf("failed to decode graph as proto text: %w, %s", tErr, err)
	} else {
		return graphDef, nil
	}

	if err := json.Unmarshal(pb, graphDef); err != nil {
		tErr = fmt.Errorf("failed to decode graph as json text: %w, %s", tErr, err)
	} else {
		return graphDef, nil
	}

	return nil, tErr
}

// SaveFile saves graph to a file based on extension
// .pb, .pbtxt, .pbtext, .json. No extension is considered
// as binary proto format equivalent to .pb extension
func SaveFile(graphDef *Def, filename string) error {
	var ext string
	var b []byte
	var err error

	parts := strings.Split(filename, ".")
	if len(parts) > 1 {
		ext = parts[len(parts)-1]
	}

	switch strings.ToLower(ext) {
	case "pb", "":
		b, err = proto.Marshal(graphDef)
		if err != nil {
			return fmt.Errorf("failed to encode graph as proto binary file: %w", err)
		}
	case "pbtxt", "pbtext":
		b, err = prototext.Marshal(graphDef)
		if err != nil {
			return fmt.Errorf("failed to encode graph as proto text file: %w", err)
		}
	case "json":
		b, err = json.MarshalIndent(graphDef, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to encode graph as json text file: %w", err)
		}
	}

	if len(b) > 0 {
		if err := os.WriteFile(filename, b, 0644); err != nil {
			return fmt.Errorf("failed to save file: %w", err)
		}

		return nil
	}

	return fmt.Errorf("failed to save file, invalid extension, pl. provide .pb, .pbtxt or .json")
}

// Import converts tensorflow Graph format to Def. In some ways
// tensorflow Graph can be thought of as higher level construct and
// Def allows looking into nodes and attributes at a raw level
func (g *Def) Import(graph *tf.Graph) error {
	bb := &bytes.Buffer{}
	bw := bufio.NewWriter(bb)

	if _, err := graph.WriteTo(bw); err != nil {
		return fmt.Errorf("failed to write to buffer: %w", err)
	}

	if err := bw.Flush(); err != nil {
		return fmt.Errorf("failed to flush buffer: %w", err)
	}

	if err := proto.Unmarshal(bb.Bytes(), g); err != nil {
		return fmt.Errorf("failed to unmarshal graph data: %w", err)
	}

	return nil
}

// Export translates Def to tensorflow Graph so it can be used
// in session runs
func (g *Def) Export(prefix string) (*tf.Graph, error) {
	pb, err := proto.Marshal(g)
	if err != nil {
		return nil, fmt.Errorf("failed to proto marshal graphdef: %w", err)
	}

	graph := tf.NewGraph()
	if err := graph.Import(pb, prefix); err != nil {
		return nil, fmt.Errorf("failed to import graph from protobuf: %w", err)
	}

	return graph, nil
}

// GetNode returns a node.Def. Each graph is essentially just a list of
// nodes and links between nodes is defined via node inputs
func (g *Def) GetNode(nodeName string) (*node.Def, error) {
	nodeDef, err := getNodeDef(g.graphDef, nodeName)
	if err != nil {
		return nil, err
	}

	return &node.Def{
		NodeDef: nodeDef,
	}, nil
}

// GetNodes returns list of nodes based on input names. Input
// names should only contain names present in the graph, otherwise
// an error is returned
func (g *Def) GetNodes(nodeNames ...string) ([]*node.Def, error) {
	existingNames := make(map[string]struct{})
	newNames := make(map[string]struct{})

	for _, nodeDef := range g.graphDef.Node {
		existingNames[nodeDef.Name] = struct{}{}
	}

	for _, name := range nodeNames {
		newNames[name] = struct{}{}
		if _, ok := existingNames[name]; !ok {
			return nil, fmt.Errorf("name %s not found in graph", name)
		}
	}

	var nodes []*node.Def
	for _, nodeDef := range g.graphDef.Node {
		if _, ok := newNames[nodeDef.Name]; ok {
			nodes = append(nodes, &node.Def{NodeDef: nodeDef})
		}
	}

	return nodes, nil
}

// Reset resets underlying graphDef proto
func (g *Def) Reset() {
	g.graphDef.Reset()
}

// DeleteNodes deletes provided node names if they exist
func (g *Def) DeleteNodes(nodeNames ...string) error {
	if len(nodeNames) == 0 {
		g.graphDef.Node = make([]*node_def_go_proto.NodeDef, 0, 0)
		return nil
	}

	existingNames := make(map[string]*node_def_go_proto.NodeDef)
	newNames := make(map[string]struct{})

	for _, nodeDef := range g.graphDef.Node {
		nodeDef := nodeDef
		existingNames[nodeDef.Name] = nodeDef
	}

	for _, name := range nodeNames {
		newNames[name] = struct{}{}
		if _, ok := existingNames[name]; !ok {
			return fmt.Errorf("name %s not found in graph", name)
		}
		delete(existingNames, name)
	}

	nodes := make([]*node_def_go_proto.NodeDef, 0, len(existingNames))
	for _, nodeDef := range existingNames {
		nodes = append(nodes, nodeDef)
	}

	g.graphDef.Node = nodes
	return nil
}

// ListNodes returns node names and filters them via options.
// Options are function literals that answer bool on each node.
// Thus, each option, if any is provided, should be satisfied
func (g *Def) ListNodes(options ...ListNodesOption) []string {
	names := make([]string, 0, len(g.graphDef.Node))
Loop:
	for _, nodeDef := range g.graphDef.Node {
		for _, option := range options {
			if !option(&node.Def{NodeDef: nodeDef}) {
				continue Loop
			}
		}
		names = append(names, nodeDef.Name)
	}

	return names
}

// ListNodesOptionOp takes op as a string and checks
// if node op matches it
func ListNodesOptionOp(op string) ListNodesOption {
	return func(node *node.Def) bool {
		return node.NodeDef.Op == op
	}
}

// ListNodesOptionWithInputs checks if each of the inputs for a node
// matches one from the input
func ListNodesOptionWithInputs(inputs ...string) ListNodesOption {
	return func(node *node.Def) bool {
		if len(inputs) == 0 && len(node.NodeDef.Input) == 0 {
			return true
		}

		inputMap := make(map[string]struct{}, len(node.NodeDef.Input))
		for _, input := range node.NodeDef.Input {
			inputMap[input] = struct{}{}
		}

		// each of the inputs must be present in the node inputs
		for _, input := range inputs {
			if _, ok := inputMap[input]; !ok {
				return false
			}
		}

		return true
	}
}

// DeleteNode deletes a node from the graph
func (g *Def) DeleteNode(nodeName string) {
	index := -1
	for i, nodeDef := range g.graphDef.Node {
		if nodeDef.Name == nodeName {
			index = i
			break
		}
	}

	if index >= 0 {
		pre := g.graphDef.Node[:index]
		post := g.graphDef.Node[index+1:]
		g.graphDef.Node = pre
		g.graphDef.Node = append(g.graphDef.Node, post...)
	}
}

// SetNodes will overwrite nodes in the graph with same name
// or append to graph if the name is not found
func (g *Def) SetNodes(nodeDefs ...*node.Def) {
	nodeMap := make(map[string]int)
	for i, nodeDef := range g.graphDef.Node {
		nodeMap[nodeDef.Name] = i
	}

	newNodeMap := make(map[string]*node.Def)
	for _, nodeDef := range nodeDefs {
		newNodeMap[nodeDef.NodeDef.Name] = nodeDef
	}

	var nodeNamesToAppend []string
	for name, nodeDef := range newNodeMap {
		if i, ok := nodeMap[name]; ok {
			g.graphDef.Node[i] = nodeDef.NodeDef
		} else {
			nodeNamesToAppend = append(nodeNamesToAppend, name)
		}
	}

	for _, name := range nodeNamesToAppend {
		nodeDef := newNodeMap[name]
		g.graphDef.Node = append(g.graphDef.Node, nodeDef.NodeDef)
	}
}

// Append appends input graphDef to the receiver thereby
// growing the node pool of receiver graph
func (g *Def) Append(graphDef *Def) error {
	existingNames := make(map[string]struct{})
	for _, nodeDef := range g.graphDef.Node {
		existingNames[nodeDef.Name] = struct{}{}
	}

	for _, nodeDef := range graphDef.graphDef.Node {
		if _, ok := existingNames[nodeDef.Name]; ok {
			return fmt.Errorf("preexisting name %s found in input graph, failed to append", nodeDef.Name)
		}
	}

	g.graphDef.Node = append(g.graphDef.Node, graphDef.graphDef.Node...)
	return nil
}

// ApplyPrefix applies input prefix to all node names and
// also changes the input strings where these nodes are being
// referred to
func (g *Def) ApplyPrefix(prefix string) error {
	b, err := proto.Marshal(g)
	if err != nil {
		return fmt.Errorf("failed to proto marshal graph: %w", err)
	}

	graph := tf.NewGraph()
	if err := graph.Import(b, prefix); err != nil {
		return fmt.Errorf("failed to import graph with prefix: %w", err)
	}

	if err := g.Import(graph); err != nil {
		return fmt.Errorf("failed to import prefixed graph: %w", err)
	}

	return nil
}

// RenameNode only changes the name of a particular node
// if it exists without affecting the inputs where such node
// name is being referred. newName should not collide with
// an existing name
func (g *Def) RenameNode(name, newName string) error {
	names := make(map[string]struct{})
	for _, nodeDef := range g.graphDef.Node {
		names[nodeDef.Name] = struct{}{}
	}

	for i, nodeDef := range g.graphDef.Node {
		if nodeDef.Name == name {
			if _, ok := names[newName]; ok {
				return fmt.Errorf("another node with name %s found in graph", newName)
			}
			nodeDef.Name = newName
			g.graphDef.Node[i] = nodeDef
			return nil
		}
	}

	return fmt.Errorf("node with name %s not found", name)
}

// newGraphDef utilizes tensorflow graph and reads it back
// in order to correctly populate version
func newGraphDef() (*graph_go_proto.GraphDef, error) {
	scope := op.NewScope()
	graph, err := scope.Finalize()
	if err != nil {
		return nil, err
	}

	bb := &bytes.Buffer{}
	bw := bufio.NewWriter(bb)

	if _, err := graph.WriteTo(bb); err != nil {
		return nil, err
	}

	if err := bw.Flush(); err != nil {
		return nil, err
	}

	graphDef := &graph_go_proto.GraphDef{}
	if err := proto.Unmarshal(bb.Bytes(), graphDef); err != nil {
		return nil, err
	}

	return graphDef, nil
}

// getNodeDef returns a NodeDef from a name
func getNodeDef(graphDef *graph_go_proto.GraphDef, name string) (*node_def_go_proto.NodeDef, error) {
	for _, nodeDef := range graphDef.Node {
		if nodeDef.Name == name {
			return nodeDef, nil
		}
	}

	return nil, fmt.Errorf("nodeDef %s not found", name)
}
