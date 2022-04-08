package tfutil

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestTensor_PrintVector(t *testing.T) {
	rand.Seed(0)

	x, err := NewTensorFromFunc(func(i int) float64 { return rand.Float64() }, 5)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(x)
}

func TestTensor_PrintMatrix(t *testing.T) {
	rand.Seed(0)

	x, err := NewTensorFromFunc(func(i int) int32 { return int32(i) }, 5, 4)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(x)
}

func TestTensor_PrintTensor2(t *testing.T) {
	rand.Seed(0)

	x, err := NewTensorFromFunc(func(i int) int32 { return int32(i) }, 5, 4, 2)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(x)
}

func TestTensor_PrintTensor(t *testing.T) {
	rand.Seed(0)

	x, err := NewTensorFromFunc(func(i int) int32 { return int32(i) }, 5, 4, 3, 2)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(x)
}
