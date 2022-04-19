package tfutil

import (
	"math/rand"
	"testing"
)

func TestTensor_ExpandDims(t *testing.T) {
	rand.Seed(0)
	f := func(int) int32 { return rand.Int31n(100) }
	x, err := NewTensorFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	if err := x.ExpandDims(0); err != nil {
		t.Fatal(err)
	}

	if !equal(x.shape, []int{1, 3, 4}) {
		t.Fatal("output shape is not as expected")
	}
}
