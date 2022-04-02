package op

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// MatrixInverse defines matrix inversion operator
func MatrixInverse(scope *op.Scope, x tf.Output) (y tf.Output) {
	if scope.Err() != nil {
		return
	}
	opspec := tf.OpSpec{
		Type: "MatrixInverse",
		Input: []tf.Input{
			x,
		},
	}
	op := scope.AddOperation(opspec)
	return op.Output(0)
}
