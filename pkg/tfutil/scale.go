package tfutil

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Scale scales receiver tensor by input value
func (g *Tensor[T]) Scale(value T) error {
	switch any(*new(T)).(type) {
	case bool, string:
		return fmt.Errorf("scale operation not supported for bool or string types")
	}

	shape := make([]int64, len(g.shape))
	for i := range shape {
		shape[i] = int64(g.shape[i])
	}

	x, err := g.Marshal()
	if err != nil {
		return fmt.Errorf("failed to get tf tensor: %w", err)
	}

	y, err := tf.NewTensor(value)
	if err != nil {
		return fmt.Errorf("failed to get tf tensor: %w", err)
	}

	root := op.NewScope()
	X := op.Placeholder(
		root.SubScope("X"),
		x.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(shape...),
		),
	)
	Y := op.Placeholder(
		root.SubScope("Y"),
		y.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(1),
		),
	)

	// define multiplication operator
	O := op.Mul(root, X, Y)

	graph, err := root.Finalize()
	if err != nil {
		return fmt.Errorf("failed to import graph: %w", err)
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
		return fmt.Errorf("failed to create new tf session: %w", err)
	}
	defer sess.Close()

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
