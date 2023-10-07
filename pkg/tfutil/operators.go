package tfutil

import (
	"fmt"

	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
)

var (
	// MatMulOp for matrix multiplication
	MatMulOp = func(scope *op.Scope, outputs ...tf.Output) (tf.Output, error) {
		if len(outputs) != 2 {
			return tf.Output{}, fmt.Errorf("operator MatMul needs len outputs = 2, got %d", len(outputs))
		}
		return op.MatMul(scope, outputs[0], outputs[1]), nil
	}

	// AbsOp for finding absolute values
	AbsOp = func(scope *op.Scope, outputs ...tf.Output) (tf.Output, error) {
		if len(outputs) != 1 {
			return tf.Output{}, fmt.Errorf("operator Abs needs len outputs = 1, got %d", len(outputs))
		}
		return op.Abs(scope, outputs[0]), nil
	}

	// RoundOp rounds values
	RoundOp = func(scope *op.Scope, outputs ...tf.Output) (tf.Output, error) {
		if len(outputs) != 1 {
			return tf.Output{}, fmt.Errorf("operator Round needs len outputs = 1, got %d", len(outputs))
		}
		return op.Round(scope, outputs[0]), nil
	}
)
