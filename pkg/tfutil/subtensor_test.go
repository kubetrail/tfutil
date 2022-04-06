package tfutil

import (
	"testing"
)

func TestTensor_Sub(t *testing.T) {
	x, err := NewFromFunc(func(i int) int32 { return int32(i) }, 2, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	s, err := x.Sub(nil, []int{2, 2, 2}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !equal(s.value, []int32{0, 1, 4, 5, 12, 13, 16, 17}) {
		t.Fatal("subtensor values not equal to expected")
	}
}
