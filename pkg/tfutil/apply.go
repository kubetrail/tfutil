package tfutil

import (
	"fmt"

	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
)

// ApplyOperators successively applies operators (last one first)
// For instance if the operator list is [op1, op2, op3],
// then it will be executed as op1(op2(op3(input)))
func ApplyOperators[T PrimitiveTypes](input *Tensor[T],
	operators ...func(*op.Scope, tf.Output) tf.Output) (*Tensor[T], error) {
	g, err := input.Clone()
	if err != nil {
		return nil, fmt.Errorf("failed to clone the tensor: %w", err)
	}

	for _, operator := range operators {
		x, err := g.Marshal()
		if err != nil {
			return nil, fmt.Errorf("failed to get tf tensor: %w", err)
		}

		root := op.NewScope()
		X := op.Placeholder(
			root.SubScope("X"),
			x.DataType(),
			op.PlaceholderShape(
				tf.MakeShape(castToInt64(g.shape)...),
			),
		)

		// define operation
		Output := operator(root, X)

		graph, err := root.Finalize()
		if err != nil {
			return nil, fmt.Errorf("failed to import graph: %w", err)
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

		if err := g.Unmarshal(out[0]); err != nil {
			return nil, fmt.Errorf("failed to unmarshal output: %w", err)
		}
	}

	return g, nil
}

// ApplyOperatorXY applies any operator accepting two inputs tensors
// and outputting single tensor, hence the name XY
func ApplyOperatorXY[T PrimitiveTypes](x, y *Tensor[T],
	operator func(*op.Scope, tf.Output, tf.Output) tf.Output) (*Tensor[T], error) {
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

	// define operation
	Output := operator(root, X, Y)

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
