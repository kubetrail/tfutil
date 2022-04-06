package tfutil

import (
	"fmt"
	"testing"
)

func TestTensor_Sub(t *testing.T) {
	x, err := NewFromFunc(func(i int) int32 { return int32(i) }, 2, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(x)

	s, err := x.Sub(nil, []int{2, 2, 2}, nil)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(s)
}
