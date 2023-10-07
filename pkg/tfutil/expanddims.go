package tfutil

import (
	"fmt"

	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
)

// ExpandDims adds a new dimension
func (tensor *Tensor[T]) ExpandDims(dim int) error {
	x, err := tensor.Marshal()
	if err != nil {
		return fmt.Errorf("failed to form tf tensor from receiver: %w", err)
	}

	y, err := tf.NewTensor(int64(dim))
	if err != nil {
		return fmt.Errorf("failed to form tf tensof for new dim: %w", err)
	}

	root := op.NewScope()
	Output := op.ExpandDims(
		root,
		op.Const(root.SubScope("x"), x),
		op.Const(root.SubScope("y"), y),
	)

	graph, err := root.Finalize()
	if err != nil {
		return fmt.Errorf("failed to import graph: %w", err)
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

	out, err := sess.Run(nil, fetches, nil)
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
