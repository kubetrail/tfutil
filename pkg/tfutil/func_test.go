package tfutil

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"testing"
)

func TestMatrixMultiplyFloat64(t *testing.T) {
	rand.Seed(0)
	f := func(int) float64 { return rand.Float64() }
	x, err := NewFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	y, err := NewFromFunc(f, 4, 2)
	if err != nil {
		t.Fatal(err)
	}

	z, err := MatrixMultiply(x, y)
	if err != nil {
		t.Fatal(err)
	}

	jb, _ := json.Marshal(z)
	expected := "{\"type\":\"tensor\",\"tfDataType\":\"Double\",\"goDataType\":\"float64\",\"shape\":[3,2],\"value\":[1.5033243536387162,0.3751283920017592,1.1535534112318055,0.7010293278394667,1.7983002735064946,1.014001106018599]}"

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

	x, err := NewFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	y, err := NewFromFunc(f, 3, 4)
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

	x, err := NewFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	y, err := NewFromFunc(f, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := MatrixMultiply(x, y); err == nil {
		t.Fatal("expected matrix multiplication to fail for string matrices")
	}
}
