package tfutil

import (
	"testing"
)

func TestDotApply(t *testing.T) {
	var tensors []*Tensor[int32]
	for i := 1; i <= 10; i++ {
		tensor, err := NewFromFunc(func(j int) int32 { return int32(i + j) }, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		tensors = append(tensors, tensor)
	}

	out, err := DotApply(
		func(values ...int32) int32 {
			sum := int32(0)
			for _, v := range values {
				sum += v
			}
			return sum
		},
		tensors...)
	if err != nil {
		t.Fatal(err)
	}

	if !equal(out.value, []int32{55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165}) {
		t.Fatal("output values do not match expected values")
	}
}
