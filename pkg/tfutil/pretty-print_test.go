package tfutil

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestTensor_PrettyPrint(t *testing.T) {
	x, err := NewFromFunc(func(i int) float64 { return rand.Float64() }, 5, 4, 3, 2)
	if err != nil {
		t.Fatal(err)
	}

	pb, err := x.PrettyPrint()
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(string(pb))
}
