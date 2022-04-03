package tfutil

import (
	"fmt"

	tfutilop "github.com/kubetrail/tfutil/pkg/op"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// MatrixInverse inverts the tensor assuming it is a square matrix
func MatrixInverse[T PrimitiveTypes](input *Tensor[T]) (*Tensor[T], error) {
	switch any(*new(T)).(type) {
	case bool, string:
		return nil, fmt.Errorf("matrix inverse operation not supported for bool or string types")
	}

	x, err := input.Marshal()
	if err != nil {
		return nil, fmt.Errorf("failed to get tf tensor: %w", err)
	}

	root := op.NewScope()
	X := op.Placeholder(
		root.SubScope("X"),
		x.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(castToInt64(input.shape)...),
		),
	)

	// operation to invert matrix.
	// needs a square matrix
	O := tfutilop.MatrixInverse(root, X)

	graph, err := root.Finalize()
	if err != nil {
		return nil, fmt.Errorf("failed to import graph: %w", err)
	}

	feeds := map[tf.Output]*tf.Tensor{
		X: x,
	}

	fetches := []tf.Output{
		O,
	}

	sess, err := tf.NewSession(
		graph,
		&tf.SessionOptions{},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create new tf session: %w", err)
	}
	defer sess.Close()

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

// Transpose transposes a tensor. perm refers to the new order
// of dimensions. For instance, if input tensor is 2x3 and perm
// for a standard transpose should be [1, 0] referring to a shape
// of 3x2. If perm values are not provides it defaults to such
// reversal of input shape.
func Transpose[T PrimitiveTypes](input *Tensor[T], perm ...int) (*Tensor[T], error) {
	x, err := input.Marshal()
	if err != nil {
		return nil, fmt.Errorf("failed to get tf tensor: %w", err)
	}

	if len(perm) == 0 {
		// perm are indices of input.shape
		perm = make([]int, len(input.shape))
		for i := range input.shape {
			perm[i] = len(input.shape) - 1 - i
		}
	} else {
		if len(perm) != len(input.shape) {
			return nil, fmt.Errorf(
				"perm len should match input shape vector length. received %d, needed %d",
				len(perm), len(input.shape),
			)
		}
	}

	y, err := tf.NewTensor(castToInt64(perm))
	if err != nil {
		return nil, fmt.Errorf("failed to get new tensor for perm vector: %w", err)
	}

	root := op.NewScope()
	X := op.Placeholder(
		root.SubScope("X"),
		x.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(castToInt64(input.shape)...),
		),
	)

	Y := op.Placeholder(
		root.SubScope("Y"),
		tf.Int64,
		op.PlaceholderShape(
			tf.MakeShape(int64(len(input.shape))),
		),
	)

	// operation to transpose a tensor.
	O := op.Transpose(root, X, Y)

	graph, err := root.Finalize()
	if err != nil {
		return nil, fmt.Errorf("failed to import graph: %w", err)
	}

	feeds := map[tf.Output]*tf.Tensor{
		X: x,
		Y: y,
	}

	fetches := []tf.Output{
		O,
	}

	sess, err := tf.NewSession(
		graph,
		&tf.SessionOptions{},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create new tf session: %w", err)
	}
	defer sess.Close()

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
