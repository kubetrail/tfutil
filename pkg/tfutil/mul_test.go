package tfutil

import (
	"fmt"
	"testing"
)

func TestTensor_Mul(t *testing.T) {
	x, err := NewTensor([]int32{1, 2, 3, 4}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	y, err := Mul(x, x)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(y)
}

func TestTensor_MulStrings(t *testing.T) {
	x, err := NewTensor([]string{"abcd", "1234", "0", "x"}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := Mul(x, x); err == nil {
		t.Fatal("expected to fail for strings")
	}
}

func TestTensor_MulBool(t *testing.T) {
	x, err := NewTensor([]bool{true, true, false, true}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := Mul(x, x); err == nil {
		t.Fatal("expected to fail for bool")
	}
}

func TestTensor_MulComplex128(t *testing.T) {
	x, err := NewTensor(
		[]complex128{
			complex(float64(2), float64(3)),
			complex(float64(4), float64(5)),
			complex(float64(6), float64(7)),
			complex(float64(8), float64(9)),
		}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(x.String())

	y, err := Mul(x, x)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(y.String())
}
