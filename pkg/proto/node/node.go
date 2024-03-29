package node

import (
	"github.com/kubetrail/tfutil/pkg/proto/attr"
	"github.com/wamuir/graft/tensorflow/core/framework/attr_value_go_proto"
	"github.com/wamuir/graft/tensorflow/core/framework/node_def_go_proto"
)

type Def struct {
	NodeDef *node_def_go_proto.NodeDef
}

// NewNodeDef returns a new instance of Def using lower level raw inputs
// such as name and op as strings and attr values
func NewNodeDef(name, op string, attrValue *attr.Value, inputs ...string) (*Def, error) {
	return &Def{
		NodeDef: &node_def_go_proto.NodeDef{
			Name:                  name,
			Op:                    op,
			Input:                 inputs,
			Device:                "",
			Attr:                  attrValue.Attr,
			ExperimentalDebugInfo: nil,
			ExperimentalType:      nil,
		},
	}, nil
}

// GetAttr returns attr value for the input keys in a node
func (g *Def) GetAttr(keys ...string) (*attr.Value, error) {
	if len(keys) == 0 {
		return &attr.Value{
			Attr: g.NodeDef.Attr,
		}, nil
	}

	keysMap := make(map[string]struct{})
	for _, key := range keys {
		keysMap[key] = struct{}{}
	}

	attrValues := make(map[string]*attr_value_go_proto.AttrValue)
	for k, v := range g.NodeDef.Attr {
		if _, ok := keysMap[k]; ok {
			attrValues[k] = v
		}
	}

	return &attr.Value{Attr: attrValues}, nil
}

// SetAttr sets attr value for a node
func (g *Def) SetAttr(attrValue *attr.Value) {
	g.NodeDef.Attr = attrValue.Attr
}

// GetDevice returns the device associated with the node
func (g *Def) GetDevice() string {
	return g.NodeDef.Device
}

// SetDevice sets the device for the node
func (g *Def) SetDevice(device string) {
	g.NodeDef.Device = device
}
