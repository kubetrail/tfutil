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
		return nil, fmt.Errorf("scale operation not supported for bool or string types")
	}

	shape := make([]int64, len(input.shape))
	for i := range shape {
		shape[i] = int64(input.shape[i])
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
			tf.MakeShape(shape...),
		),
	)

	// operation to invert matrix.
	// needs a square matrix
	O := tfutilop.MatrixInverse(root, X)

	graph, err := root.Finalize()
	if err != nil {
		return nil, fmt.Errorf("failed to import graph: %w", err)
	}

	// prepare data feed specifying names of the operation.
	// names x and dim come from python code, see def of reshape
	// function taking inputs x and dim
	feeds := map[tf.Output]*tf.Tensor{
		X: x,
	}

	// prepare data outputs from tensorflow run.
	// Identity is the final output point of the graph.
	fetches := []tf.Output{
		O,
	}

	// start new session
	sess, err := tf.NewSession(
		graph,
		&tf.SessionOptions{},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create new tf session: %w", err)
	}
	defer sess.Close()

	// run session feeding feeds and fetching fetches
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
