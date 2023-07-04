package tfutil

import (
	"fmt"

	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
)

// Mul performs element wise multiplication of two tensors
func Mul[T PrimitiveTypes](x, y *Tensor[T]) (*Tensor[T], error) {
	xTfTensor, err := x.Marshal()
	if err != nil {
		return nil, fmt.Errorf("failed to get tf tensor: %w", err)
	}

	yTfTensor, err := y.Marshal()
	if err != nil {
		return nil, fmt.Errorf("failed to get tf tensor: %w", err)
	}

	root := op.NewScope()
	X := op.Placeholder(
		root.SubScope("X"),
		xTfTensor.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(castToInt64(x.shape)...),
		),
	)
	Y := op.Placeholder(
		root.SubScope("Y"),
		yTfTensor.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(castToInt64(y.shape)...),
		),
	)

	// define multiplication operator
	Output := op.Mul(root, X, Y)

	graph, err := root.Finalize()
	if err != nil {
		return nil, fmt.Errorf("failed to import graph: %w", err)
	}

	feeds := map[tf.Output]*tf.Tensor{
		X: xTfTensor,
		Y: yTfTensor,
	}

	fetches := []tf.Output{
		Output,
	}

	sess, err := tf.NewSession(
		graph,
		&tf.SessionOptions{},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create new tf session: %w", err)
	}
	defer func(sess *tf.Session) {
		err := sess.Close()
		if err != nil {
			panic(err)
		}
	}(sess)

	out, err := sess.Run(feeds, fetches, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to run tf session: %w", err)
	}

	if len(out) != 1 {
		return nil, fmt.Errorf("expected session run output to have length 1, got %d", len(out))
	}

	output := &Tensor[T]{}
	if err := output.Unmarshal(out[0]); err != nil {
		return nil, fmt.Errorf("failed to unmarshal output: %w", err)
	}

	return output, nil
}
