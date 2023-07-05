package tfutil

import (
	"fmt"

	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
)

// Reshape reshapes to new shape
func (g *Tensor[T]) Reshape(shape ...int) error {
	x, err := g.Marshal()
	if err != nil {
		return fmt.Errorf("failed to get tf tensor: %w", err)
	}

	y, err := tf.NewTensor(castToInt64(shape))
	if err != nil {
		return fmt.Errorf("failed to get tf tensor: %w", err)
	}

	root := op.NewScope()
	X := op.Placeholder(
		root.SubScope("X"),
		x.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(castToInt64(g.shape)...),
		),
	)
	Y := op.Placeholder(
		root.SubScope("Y"),
		y.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(int64(len(shape))),
		),
	)

	// define reshape operation
	Output := op.Reshape(root, X, Y)

	graph, err := root.Finalize()
	if err != nil {
		return fmt.Errorf("failed to import graph: %w", err)
	}

	feeds := map[tf.Output]*tf.Tensor{
		X: x,
		Y: y,
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

	if err := g.Unmarshal(out[0]); err != nil {
		return fmt.Errorf("failed to unmarshal output: %w", err)
	}

	return nil
}
