package tfutil

import "fmt"

func shapeToint64Shape(shape []int) []int64 {
	newShape := make([]int64, len(shape))
	for i := range shape {
		newShape[i] = int64(shape[i])
	}

	return newShape
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
