package tfutil

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"testing"
)

func TestApplyOperators(t *testing.T) {
	// create a random square matrix using a generator func
	x, err := NewTensorFromFunc(
		func(int) float64 { return rand.Float64() }, 5, 5)
	if err != nil {
		t.Fatal(err)
	}

	y, err := MatrixInverse(x)
	if err != nil {
		t.Fatal(err)
	}

	z, err := MatrixMultiply(x, y)
	if err != nil {
		t.Fatal(err)
	}

	if err := z.Apply(RoundOp); err != nil {
		t.Fatal(err)
	}

	if err := z.Apply(AbsOp); err != nil {
		t.Fatal(err)
	}

	jb, err := json.Marshal(z)
	if err != nil {
		t.Fatal(err)
	}

	expected := `{"type":"tensor","tfDataType":"Double","goDataType":"float64","shape":[5,5],"value":[1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1]}`

	if !bytes.Equal(jb, []byte(expected)) {
		t.Fatal("output does not match expected value")
	}
}

func TestApplyDualInputOperator(t *testing.T) {
	// create a random square matrix using a generator func
	x, err := NewTensorFromFunc(
		func(int) float64 { return rand.Float64() }, 5, 5)
	if err != nil {
		t.Fatal(err)
	}

	y, err := MatrixInverse(x)
	if err != nil {
		t.Fatal(err)
	}

	z, err := Apply(MatMulOp, x, y)
	if err != nil {
		t.Fatal(err)
	}

	if err := z.Apply(RoundOp); err != nil {
		t.Fatal(err)
	}

	if err := z.Apply(AbsOp); err != nil {
		t.Fatal(err)
	}

	jb, err := json.Marshal(z)
	if err != nil {
		t.Fatal(err)
	}

	expected := "{\"type\":\"tensor\",\"tfDataType\":\"Double\",\"goDataType\":\"float64\",\"shape\":[5,5],\"value\":[1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1]}"
	fmt.Println("***", expected)

	if !bytes.Equal(jb, []byte(expected)) {
		t.Fatal("output does not match expected value")
	}
}
