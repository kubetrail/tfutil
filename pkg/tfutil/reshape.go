package tfutil

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func shapeToint64Shape(shape []int) []int64 {
	newShape := make([]int64, len(shape))
	for i := range shape {
		newShape[i] = int64(shape[i])
	}

	return newShape
}

// Reshape reshapes to new shape
func (g *Tensor[T]) Reshape(shape ...int) error {
	x, err := g.Marshal()
	if err != nil {
		return fmt.Errorf("failed to get tf tensor: %w", err)
	}

	y, err := tf.NewTensor(shapeToint64Shape(shape))
	if err != nil {
		return fmt.Errorf("failed to get tf tensor: %w", err)
	}

	root := op.NewScope()
	X := op.Placeholder(
		root.SubScope("X"),
		x.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(shapeToint64Shape(g.shape)...),
		),
	)
	Y := op.Placeholder(
		root.SubScope("Y"),
		y.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(int64(len(shape))),
		),
	)

	// Define the operation node that accepts A & x as inputs
	product := op.Reshape(root, X, Y)

	graph, err := root.Finalize()
	if err != nil {
		return fmt.Errorf("failed to import graph: %w", err)
	}

	// prepare data feed specifying names of the operation.
	// names x and dim come from python code, see def of reshape
	// function taking inputs x and dim
	feeds := map[tf.Output]*tf.Tensor{
		X: x,
		Y: y,
	}

	// prepare data outputs from tensorflow run.
	// Identity is the final output point of the graph.
	fetches := []tf.Output{
		product,
	}

	// start new session
	sess, err := tf.NewSession(
		graph,
		&tf.SessionOptions{},
	)
	if err != nil {
		return fmt.Errorf("failed to create new tf session: %w", err)
	}
	defer sess.Close()

	// run session feeding feeds and fetching fetches
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
