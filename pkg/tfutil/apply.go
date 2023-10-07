package tfutil

import (
	"fmt"

	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
)

// Apply applies operator on the receiver tensor modifying it. For instance,
// MatrixInverseOp is an operator that will invert the matrix assuming that
// the receiver tensor is a matrix that can be inverted.
func (tensor *Tensor[T]) Apply(operator Operator) error {
	x, err := tensor.Marshal()
	if err != nil {
		return fmt.Errorf("failed to get tf tensor: %w", err)
	}

	root := op.NewScope()
	X := op.Placeholder(
		root.SubScope("X"),
		x.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(castToInt64(tensor.shape)...),
		),
	)

	// define operation
	Output, err := operator(root, X)
	if err != nil {
		return fmt.Errorf("invalid operator: %w", err)
	}

	graph, err := root.Finalize()
	if err != nil {
		return fmt.Errorf("failed to import graph: %w", err)
	}

	feeds := map[tf.Output]*tf.Tensor{
		X: x,
	}

	fetches := []tf.Output{
		Output,
	}

	sess, err := tf.NewSession(
		graph,
		&tf.SessionOptions{},
	)
	if err != nil {
		return fmt.Errorf("failed to create new tf session: %w", err)
	}
	defer func(sess *tf.Session) {
		err := sess.Close()
		if err != nil {
			panic(err)
		}
	}(sess)

	out, err := sess.Run(feeds, fetches, nil)
	if err != nil {
		return fmt.Errorf("failed to run tf session: %w", err)
	}

	if len(out) != 1 {
		return fmt.Errorf("expected session run output to have length 1, got %d", len(out))
	}

	if err := tensor.Unmarshal(out[0]); err != nil {
		return fmt.Errorf("failed to unmarshal output: %w", err)
	}

	return nil
}

// Apply applies an operator accepting a variadic input argument of
// tensors. This is useful for things such as matrix multiplication that
// require two input matrices
func Apply[T PrimitiveTypes](operator Operator, tensors ...*Tensor[T]) (*Tensor[T], error) {
	root := op.NewScope()
	outputs := make([]tf.Output, len(tensors))
	feeds := make(map[tf.Output]*tf.Tensor)
	for i, tensor := range tensors {
		t, err := tensor.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to get tf tensor: %w", err)
		}

		outputs[i] = op.Placeholder(
			root.SubScope(fmt.Sprintf("T%d", i)),
			t.DataType(),
			op.PlaceholderShape(
				tf.MakeShape(castToInt64(tensor.shape)...),
			),
		)
		feeds[outputs[i]] = t
	}

	// define operation
	Output, err := operator(root, outputs...)
	if err != nil {
		return nil, fmt.Errorf("invalid operator: %w", err)
	}

	graph, err := root.Finalize()
	if err != nil {
		return nil, fmt.Errorf("failed to import graph: %w", err)
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
