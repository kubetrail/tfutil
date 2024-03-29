package tfutil

import (
	"fmt"

	"golang.org/x/exp/constraints"
)

// castToInt64 is a generic function that can cast input
// slice of integers to a slice of int64
func castToInt64[T constraints.Integer](shape []T) []int64 {
	newShape := make([]int64, len(shape))
	for i, v := range shape {
		newShape[i] = int64(v)
	}

	return newShape
}

// equal check element by element equality of two slices of
// same type
func equal[T PrimitiveTypes | int](x, y []T) bool {
	if len(x) != len(y) {
		return false
	}

	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}

	return true
}

func clone[T PrimitiveTypes | int](x []T) []T {
	y := make([]T, len(x))
	copy(y, x)
	return y
}

// numElements is the total number of elements represented by
// shape slice.
// error is returned if shape value is negative or zero.
func numElements(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, nil
	}

	n := shape[0]
	if n <= 0 {
		return -1, fmt.Errorf("please provide positive shape values")
	}

	if len(shape) == 1 {
		return n, nil
	}

	for i := 1; i < len(shape); i++ {
		if shape[i] <= 0 {
			return -1, fmt.Errorf("please provide positive shape values")
		}
		n *= shape[i]
	}

	return n, nil
}
