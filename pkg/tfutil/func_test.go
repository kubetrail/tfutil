package tfutil

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"testing"
)

func TestMatrixMultiplyFloat64(t *testing.T) {
	rand.Seed(0)
	f := func(int) float64 { return float64(rand.Intn(100)) / 100 }

	x, err := NewTensorFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	y, err := NewTensorFromFunc(f, 4, 2)
	if err != nil {
		t.Fatal(err)
	}

	z, err := MatrixMultiply(x, y)
	if err != nil {
		t.Fatal(err)
	}

	jb, _ := json.Marshal(z)
	expected := "{\"type\":\"tensor\",\"tfDataType\":\"Double\",\"goDataType\":\"float64\",\"shape\":[3,2],\"value\":[0.5977,0.713,1.0464,1.3804,0.9746,1.277]}"

	if !bytes.Equal(jb, []byte(expected)) {
		t.Fatal("output does not match expected output")
	}
}

func TestMatrixMultiplyInt32(t *testing.T) {
	rand.Seed(0)
	f := func(int) int32 { return rand.Int31n(100) }
	x, err := NewTensorFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	y, err := NewTensorFromFunc(f, 4, 2)
	if err != nil {
		t.Fatal(err)
	}

	z, err := MatrixMultiply(x, y)
	if err != nil {
		t.Fatal(err)
	}

	jb, _ := json.Marshal(z)
	expected := "{\"type\":\"tensor\",\"tfDataType\":\"Int32\",\"goDataType\":\"int32\",\"shape\":[3,2],\"value\":[5977,7130,10464,13804,9746,12770]}"

	if !bytes.Equal(jb, []byte(expected)) {
		t.Fatal("output of matrix multiplication does not match expected output")
	}
}

func TestMatrixMultiplyBool(t *testing.T) {
	rand.Seed(0)
	f := func(int) bool {
		if rand.Float64() > 0.5 {
			return true
		}
		return false
	}

	x, err := NewTensorFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	y, err := NewTensorFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := MatrixMultiply(x, y); err == nil {
		t.Fatal("expected matrix multiplication to fail for bool matrices")
	}
}

func TestMatrixMultiplyString(t *testing.T) {
	rand.Seed(0)
	f := func(int) string {
		return string([]byte{byte(rand.Intn(26) + 'a')})
	}

	x, err := NewTensorFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	y, err := NewTensorFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := MatrixMultiply(x, y); err == nil {
		t.Fatal("expected matrix multiplication to fail for string matrices")
	}
}

func TestCast(t *testing.T) {
	rand.Seed(0)
	f := func(int) int32 { return rand.Int31n(100) }
	x, err := NewTensorFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	y := &Tensor[float64]{}
	if err := Cast(x, y); err != nil {
		t.Fatal(err)
	}

	for i := range x.value {
		if float64(x.value[i]) != y.value[i] {
			t.Fatal("output does not match input")
		}
	}
}
