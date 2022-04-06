package tfutil

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Sub fetches sub a new tensor without altering original.
// startIndices if nil is set to a slice
// of zeros indicating starting from the beginning of the tensor. Length of
// such slice is always equal to the receiver dimension indicating start
// value for each dimension.
//Similarly, endLengths indicate end value similar
// to how a sub slicing end index works. if endLengths is nil, it is set
// to the shape slice of the receiver tensor.
//strides are jumps and if nil is set to slice of ones.
// The lengths of each of these inputs is, therefore, either nil or
// equal to the length of the shape of the receiver tensor
func (g *Tensor[T]) Sub(start, end, stride []int) (*Tensor[T], error) {
	if start == nil {
		start = make([]int, len(g.shape))
	}
	if end == nil {
		end = g.shape
	}
	if stride == nil {
		stride = make([]int, len(g.shape))
		for i := range stride {
			stride[i] = 1
		}
	}

	if len(start) != len(g.shape) ||
		len(end) != len(g.shape) ||
		len(stride) != len(g.shape) {
		return nil, fmt.Errorf("inputs should either be nil or have lengths equal to %d", len(g.shape))
	}

	x, err := g.Marshal()
	if err != nil {
		return nil, fmt.Errorf("failed to get tf tensor: %w", err)
	}

	startTensor, err := tf.NewTensor(castToInt64(start))
	if err != nil {
		return nil, fmt.Errorf("failed to get tf tensor: %w", err)
	}

	endTensor, err := tf.NewTensor(castToInt64(end))
	if err != nil {
		return nil, fmt.Errorf("failed to get tf tensor: %w", err)
	}

	strideTensor, err := tf.NewTensor(castToInt64(stride))
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
	Start := op.Placeholder(
		root.SubScope("Start"),
		startTensor.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(int64(len(start))),
		),
	)
	End := op.Placeholder(
		root.SubScope("End"),
		startTensor.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(int64(len(end))),
		),
	)
	Stride := op.Placeholder(
		root.SubScope("Stride"),
		startTensor.DataType(),
		op.PlaceholderShape(
			tf.MakeShape(int64(len(stride))),
		),
	)

	// define operation
	Output := op.StridedSlice(root, X, Start, End, Stride)

	graph, err := root.Finalize()
	if err != nil {
		return nil, fmt.Errorf("failed to import graph: %w", err)
	}

	feeds := map[tf.Output]*tf.Tensor{
		X:      x,
		Start:  startTensor,
		End:    endTensor,
		Stride: strideTensor,
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
