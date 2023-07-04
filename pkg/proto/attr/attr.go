package attr

import (
	"github.com/wamuir/graft/tensorflow/core/framework/attr_value_go_proto"
)

type Value struct {
	Attr map[string]*attr_value_go_proto.AttrValue
}

// NewAttrValue creates a new instance of attribute value where
// input value can be one of: *tf.Tensor, *attr_value_go_proto.AttrValue
// tf.DataType, []int for shape or []int64 for list of values
func NewAttrValue() *Value {
	return &Value{Attr: map[string]*attr_value_go_proto.AttrValue{}}
}
