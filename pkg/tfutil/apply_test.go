package tfutil

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/tensorflow/tensorflow/tensorflow/go/op"
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

	// apply transformation operators in sequence with
	// last one being applied first. In this case
	// the output will be from abs(round(z))
	z, err = ApplyOperators(z, op.Abs, op.Round)
	if err != nil {
		t.Fatal(err)
	}

	jb, err := json.Marshal(z)
	if err != nil {
		t.Fatal(err)
	}

	expected := "{\"type\":\"tensor\",\"tfDataType\":\"Double\",\"goDataType\":\"float64\",\"shape\":[5,5],\"value\":[1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1]}"

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

	z, err := ApplyOperatorXY(x, y,
		func(scope *op.Scope, x, y tf.Output) tf.Output {
			return op.MatMul(scope, x, y)
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	// apply transformation operators in sequence with
	// last one being applied first. In this case
	// the output will be from abs(round(z))
	z, err = ApplyOperators(z, op.Abs, op.Round)
	if err != nil {
		t.Fatal(err)
	}

	jb, err := json.Marshal(z)
	if err != nil {
		t.Fatal(err)
	}

	expected := "{\"type\":\"tensor\",\"tfDataType\":\"Double\",\"goDataType\":\"float64\",\"shape\":[5,5],\"value\":[1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1]}"

	if !bytes.Equal(jb, []byte(expected)) {
		t.Fatal("output does not match expected value")
	}
}
