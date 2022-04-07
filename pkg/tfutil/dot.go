package tfutil

import "fmt"

// DotApply applies input function f over each corresponding elements of input tensors
// and returns a new tensor using output of that function. For instance, if input function
// f sums up all values of its input, then this will have a result of performing element
// wise sum over input tensors
func DotApply[T PrimitiveTypes](f func(values ...T) T, tensors ...*Tensor[T]) (*Tensor[T], error) {
	if len(tensors) == 0 {
		return nil, nil
	}

	shape := tensors[0].shape
	for _, tensor := range tensors {
		if !equal(shape, tensor.shape) {
			return nil, fmt.Errorf("all tensors must be of same shape")
		}
	}

	value := make([]T, len(tensors[0].value))
	for i := range tensors[0].value {
		values := make([]T, len(tensors))
		for j, tensor := range tensors {
			values[j] = tensor.value[i]
		}

		value[i] = f(values...)
	}

	return NewTensor(value, shape...)
}
