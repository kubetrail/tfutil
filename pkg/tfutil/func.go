package tfutil

import (
	"fmt"
	"math/cmplx"

	tfutilop "github.com/kubetrail/tfutil/pkg/op"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// MatrixInverse inverts the tensor assuming it is a square matrix,
// otherwise it will throw error
func MatrixInverse[T PrimitiveTypes](input *Tensor[T]) (*Tensor[T], error) {
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
	Output := tfutilop.MatrixInverse(root, X)

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

	output := &Tensor[T]{}
	if err := output.Unmarshal(out[0]); err != nil {
		return nil, fmt.Errorf("failed to unmarshal output: %w", err)
	}

	return output, nil
}

// MatrixMultiply performs matrix multiplication
func MatrixMultiply[T PrimitiveTypes](x, y *Tensor[T]) (*Tensor[T], error) {
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

	// operation to invert matrix.
	// needs a square matrix
	Output := op.MatMul(root, X, Y)

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

// Round rounds the elements of input matrix
func Round[T PrimitiveTypes](input *Tensor[T]) (*Tensor[T], error) {
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
	Output := op.Round(root, X)

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
	Output := op.Transpose(root, X, Y)

	graph, err := root.Finalize()
	if err != nil {
		return nil, fmt.Errorf("failed to import graph: %w", err)
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

// NewFromFunc generates a new tensor using an input function that is called for
// each element
func NewFromFunc[T PrimitiveTypes](f func(int) T, shape ...int) (*Tensor[T], error) {
	n, err := numElements(shape)
	if err != nil {
		return nil, fmt.Errorf("invalid shape: %w", err)
	}

	values := make([]T, n)
	for i := range values {
		values[i] = f(i)
	}

	return NewTensor(values, shape...)
}

// Complex128 packs input real and imaginary parts to a complex128 valued tensor
func Complex128(realT, imagT *Tensor[float64]) (*Tensor[complex128], error) {
	if realT == nil || imagT == nil {
		return nil, fmt.Errorf("inputs can't be nil")
	}

	if len(realT.shape) != len(imagT.shape) {
		return nil, fmt.Errorf("input tensor shape lengths do not match")
	}

	for i := range realT.shape {
		if realT.shape[i] != imagT.shape[i] {
			return nil, fmt.Errorf("input tensor shape values do not match")
		}
	}

	if len(realT.value) != len(imagT.value) {
		return nil, fmt.Errorf("input tensor value lengths do not match")
	}

	c := make([]complex128, len(realT.value))
	for i := range realT.value {
		c[i] = complex(realT.value[i], imagT.value[i])
	}

	return NewTensor(c, realT.shape...)
}

// Complex64 packs input real and imaginary parts to a complex64 valued tensor
func Complex64(realT, imagT *Tensor[float32]) (*Tensor[complex64], error) {
	if realT == nil || imagT == nil {
		return nil, fmt.Errorf("inputs can't be nil")
	}

	if len(realT.shape) != len(imagT.shape) {
		return nil, fmt.Errorf("input tensor shape lengths do not match")
	}

	for i := range realT.shape {
		if realT.shape[i] != imagT.shape[i] {
			return nil, fmt.Errorf("input tensor shape values do not match")
		}
	}

	if len(realT.value) != len(imagT.value) {
		return nil, fmt.Errorf("input tensor value lengths do not match")
	}

	c := make([]complex64, len(realT.value))
	for i := range realT.value {
		c[i] = complex(realT.value[i], imagT.value[i])
	}

	return NewTensor(c, realT.shape...)
}

// Real64 pulls real elements from input tensor and packs
// them into a new tensor of float64 type
func Real64(complexT *Tensor[complex128]) *Tensor[float64] {
	values := make([]float64, len(complexT.value))
	for i := range values {
		values[i] = real(complexT.value[i])
	}

	realT, _ := NewTensor(values, complexT.shape...)
	return realT
}

// Imag64 pulls imaginary elements from input tensor and packs
// them into a new tensor of float64 type
func Imag64(complexT *Tensor[complex128]) *Tensor[float64] {
	values := make([]float64, len(complexT.value))
	for i := range values {
		values[i] = imag(complexT.value[i])
	}

	realT, _ := NewTensor(values, complexT.shape...)
	return realT
}

// Real32 pulls real elements from input tensor and packs
// them into a new tensor of float32 type
func Real32(complexT *Tensor[complex64]) *Tensor[float32] {
	values := make([]float32, len(complexT.value))
	for i := range values {
		values[i] = real(complexT.value[i])
	}

	realT, _ := NewTensor(values, complexT.shape...)
	return realT
}

// Imag32 pulls imaginary elements from input tensor and packs
// them into a new tensor of float32 type
func Imag32(complexT *Tensor[complex64]) *Tensor[float32] {
	values := make([]float32, len(complexT.value))
	for i := range values {
		values[i] = imag(complexT.value[i])
	}

	realT, _ := NewTensor(values, complexT.shape...)
	return realT
}

// Abs returns absolute valued tensor such that each element
// of output is absolute value of each of the complex values
// of input
func Abs(complexT *Tensor[complex128]) *Tensor[float64] {
	f := func(i int) float64 {
		return cmplx.Abs(complexT.value[i])
	}

	absT, _ := NewFromFunc(f, complexT.shape...)
	return absT
}
