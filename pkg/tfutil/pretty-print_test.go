package tfutil

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestTensor_PrettyPrintVector(t *testing.T) {
	rand.Seed(0)

	x, err := NewFromFunc(func(i int) float64 { return rand.Float64() }, 5)
	if err != nil {
		t.Fatal(err)
	}

	pb, err := x.PrettyPrint()
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(string(pb))
}

func TestTensor_PrettyPrintMatrix(t *testing.T) {
	rand.Seed(0)

	x, err := NewFromFunc(func(i int) int32 { return int32(i) }, 5, 4)
	if err != nil {
		t.Fatal(err)
	}

	pb, err := x.PrettyPrint()
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(string(pb))
}

func TestTensor_PrettyPrintTensor2(t *testing.T) {
	rand.Seed(0)

	x, err := NewFromFunc(func(i int) int32 { return int32(i) }, 5, 4, 2)
	if err != nil {
		t.Fatal(err)
	}

	pb, err := x.PrettyPrint()
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(string(pb))
}

func TestTensor_PrettyPrintTensor(t *testing.T) {
	rand.Seed(0)

	x, err := NewFromFunc(func(i int) int32 { return int32(i) }, 5, 4, 3, 2)
	if err != nil {
		t.Fatal(err)
	}

	pb, err := x.PrettyPrint()
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(string(pb))
}
