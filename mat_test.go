package tfutil

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestNewScalarFloat64(t *testing.T) {
	tfTensor, err := tf.NewTensor(float64(3.14))
	if err != nil {
		t.Fatal(err)
	}

	scalar := &Scalar[float64]{}
	if err := scalar.Unmarshal(tfTensor); err != nil {
		t.Fatal(err)
	}

	if scalar.Value() != tfTensor.Value().(float64) {
		t.Fatal("did not get expected value")
	}
}

func TestNewScalarString(t *testing.T) {
	tfTensor, err := tf.NewTensor("test")
	if err != nil {
		t.Fatal(err)
	}

	scalar := &Scalar[string]{}
	if err := scalar.Unmarshal(tfTensor); err != nil {
		t.Fatal(err)
	}

	if scalar.Value() != tfTensor.Value().(string) {
		t.Fatal("did not get expected value")
	}
}

func TestNewScalarBool(t *testing.T) {
	tfTensor, err := tf.NewTensor(true)
	if err != nil {
		t.Fatal(err)
	}

	scalar := &Scalar[bool]{}
	if err := scalar.Unmarshal(tfTensor); err != nil {
		t.Fatal(err)
	}

	if scalar.Value() != tfTensor.Value().(bool) {
		t.Fatal("did not get expected value")
	}
}

func TestNewScalarComplex128(t *testing.T) {
	tfTensor, err := tf.NewTensor(complex(float64(3.14), float64(2.71)))
	if err != nil {
		t.Fatal(err)
	}

	scalar := &Scalar[complex128]{}
	if err := scalar.Unmarshal(tfTensor); err != nil {
		t.Fatal(err)
	}

	if scalar.Value() != tfTensor.Value().(complex128) {
		t.Fatal("did not get expected value")
	}
}

func TestNewScalarComplex64(t *testing.T) {
	tfTensor, err := tf.NewTensor(complex(float32(3.14), float32(2.71)))
	if err != nil {
		t.Fatal(err)
	}

	scalar := &Scalar[complex64]{}
	if err := scalar.Unmarshal(tfTensor); err != nil {
		t.Fatal(err)
	}

	if scalar.Value() != tfTensor.Value().(complex64) {
		t.Fatal("did not get expected value")
	}
}

func TestNewScalarUint8(t *testing.T) {
	tfTensor, err := tf.NewTensor(uint8(3))
	if err != nil {
		t.Fatal(err)
	}

	scalar := &Scalar[byte]{}
	if err := scalar.Unmarshal(tfTensor); err != nil {
		t.Fatal(err)
	}

	if scalar.Value() != tfTensor.Value().(byte) {
		t.Fatal("did not get expected value")
	}
}

func TestNewTensorUint8(t *testing.T) {
	expected := []byte{1, 2, 3, 4}
	tfTensor, err := tf.NewTensor([][]byte{{expected[0], expected[1]}, {expected[2], expected[3]}})
	if err != nil {
		t.Fatal(err)
	}

	tensor := &Tensor[uint8]{}
	if err := tensor.Unmarshal(tfTensor); err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(tensor.Value(), expected) {
		t.Fatal("did not get expected value")
	}
}

func TestNewTensorStrings(t *testing.T) {
	expected := []string{"ab", "cde", "12334", "%$#@!*()"}
	tfTensor, err := tf.NewTensor([][]string{{expected[0], expected[1]}, {expected[2], expected[3]}})
	if err != nil {
		t.Fatal(err)
	}

	tensor := &Tensor[string]{}
	if err := tensor.Unmarshal(tfTensor); err != nil {
		t.Fatal(err)
	}

	got := tensor.Value()
	if len(got) != len(expected) {
		t.Fatal("lengths of expected and got slices do not match")
	}

	for i := range got {
		if got[i] != expected[i] {
			t.Fatal("expected", expected[i], ", got", got[i])
		}
	}
}

func TestNewTensorFloat64(t *testing.T) {
	expected := []float64{0.1, 0.2, 0.3, 0.4}
	tfTensor, err := tf.NewTensor([][]float64{{expected[0], expected[1]}, {expected[2], expected[3]}})
	if err != nil {
		t.Fatal(err)
	}

	tensor := &Tensor[float64]{}
	if err := tensor.Unmarshal(tfTensor); err != nil {
		t.Fatal(err)
	}

	got := tensor.Value()
	if len(got) != len(expected) {
		t.Fatal("lengths of expected and got slices do not match")
	}

	for i := range got {
		if got[i] != expected[i] {
			t.Fatal("expected", expected[i], ", got", got[i])
		}
	}
}

func TestTfTensorValueShapeOfScalar(t *testing.T) {
	tfTensor, err := tf.NewTensor(float32(3.14))
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println("shape:", tfTensor.Shape())
	fmt.Println("value:", tfTensor.Value())
	fmt.Println("dataType:", tfTensor.DataType())
}

func TestTensor_GetElement(t *testing.T) {
	expected := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9}
	tensor, err := NewTensor(expected, 3, 3)
	if err != nil {
		t.Fatal(err)
	}

	count := 0
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			e, err := tensor.GetElement(i, j)
			if err != nil {
				t.Fatal(err)
			}

			if e != expected[count] {
				t.Fatal("expected", expected[count], ", got", e, ", at indices", []int{i, j})
			}

			count++
		}
	}
}

func TestTensor_GetElement2(t *testing.T) {
	ni, nj, nk, nl := 2, 3, 4, 5
	expected := make([]float64, ni*nj*nk*nl)
	for i := range expected {
		expected[i] = rand.Float64()
	}

	tensor, err := NewTensor(expected, ni, nj, nk, nl)
	if err != nil {
		t.Fatal(err)
	}

	count := 0
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				for l := 0; l < 5; l++ {
					e, err := tensor.GetElement(i, j, k, l)
					if err != nil {
						t.Fatal(err)
					}

					if e != expected[count] {
						t.Fatal("expected", expected[count], ", got", e, ", at indices", []int{i, j, k, l})
					}

					count++
				}
			}
		}
	}
}

func TestTensor_GetElement3(t *testing.T) {
	ni, nj, nk, nl := 2, 3, 4, 5
	expected := make([]float64, ni*nj*nk*nl)
	for i := range expected {
		expected[i] = rand.Float64()
	}

	tensor, err := NewTensor(expected, ni, nj, nk, nl)
	if err != nil {
		t.Fatal(err)
	}

	tfTensor, err := tensor.Marshal()
	if err != nil {
		t.Fatal(err)
	}

	expectedTensor, ok := tfTensor.Value().([][][][]float64)
	if !ok {
		t.Fatal("invalid type assertion")
	}

	count := 0
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				for l := 0; l < 5; l++ {
					e, err := tensor.GetElement(i, j, k, l)
					if err != nil {
						t.Fatal(err)
					}

					if e != expectedTensor[i][j][k][l] {
						t.Fatal("expected", expected[count], ", got", e, ", at indices", []int{i, j, k, l})
					}

					count++
				}
			}
		}
	}
}

func TestTensor_String(t *testing.T) {
	ni, nj := 2, 3
	expected := make([]float64, ni*nj)
	for i := range expected {
		expected[i] = rand.Float64()
	}

	tensor, err := NewTensor(expected, ni, nj)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(tensor)

	jb, err := json.Marshal(tensor)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(string(jb))
}

func TestScalar_String(t *testing.T) {
	scalar := NewScalar(3.14)

	fmt.Println(scalar)

	jb, err := json.Marshal(scalar)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(string(jb))
}
