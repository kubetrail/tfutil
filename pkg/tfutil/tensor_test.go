package tfutil

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
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

func TestTensor_SetElement(t *testing.T) {
	ni, nj, nk, nl := 2, 3, 4, 5
	expected := make([]float64, ni*nj*nk*nl)

	tensor, err := NewTensor(expected, ni, nj, nk, nl)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				for l := 0; l < 5; l++ {
					if err := tensor.SetElement(rand.Float64(), i, j, k, l); err != nil {
						t.Fatal(err)
					}
				}
			}
		}
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

func TestTensor_Scale(t *testing.T) {
	x, err := NewTensor([]int32{1, 2, 3, 4}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	if err := x.Scale(int32(2)); err != nil {
		t.Fatal(err)
	}

	fmt.Println(x)
}

func TestTensor_ScaleStrings(t *testing.T) {
	x, err := NewTensor([]string{"abcd", "1234", "0", "x"}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	if err := x.Scale("zzz"); err == nil {
		t.Fatal("expected to fail for strings")
	}
}

func TestTensor_ScaleBool(t *testing.T) {
	x, err := NewTensor([]bool{true, true, false, true}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	if err := x.Scale(false); err == nil {
		t.Fatal("expected to fail for bool")
	}
}

func TestTensor_ScaleComplex128(t *testing.T) {
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

	fmt.Println(x)

	if err := x.Scale(complex(float64(1.2), float64(2.3))); err != nil {
		t.Fatal(err)
	}

	fmt.Println(x)
}

func TestTensor_Reshape(t *testing.T) {
	m, err := NewTensor([]int32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	if err := m.Reshape(3, 2); err != nil {
		t.Fatal(err)
	}

	fmt.Println(m)
}

func TestTensor_ReshapeString(t *testing.T) {
	m, err := NewTensor([]string{"abcd", "1234", "c", "d0", "ee", "ffff"}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	if err := m.Reshape(3, 2); err != nil {
		t.Fatal(err)
	}

	fmt.Println(m)
}

func TestTensor_ReshapeBool(t *testing.T) {
	m, err := NewTensor([]bool{true, true, false, true, false, false}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	if err := m.Reshape(3, 2); err != nil {
		t.Fatal(err)
	}

	fmt.Println(m)
}

func TestTensor_Inv(t *testing.T) {
	m, err := NewTensor(make([]float64, 25), 5, 5)
	if err != nil {
		t.Fatal(err)
	}

	for i := range m.value {
		m.value[i] = rand.Float64()
	}

	fmt.Println(m)

	output, err := MatrixInverse(m)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(output)
}

func TestMatrixInverse(t *testing.T) {
	m, err := NewTensor(make([]float32, 25), 5, 5)
	if err != nil {
		t.Fatal(err)
	}

	for i := range m.value {
		m.value[i] = rand.Float32()
	}

	fmt.Println(m)

	output, err := MatrixInverse(m)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(output)
}

func TestMatrixTranspose(t *testing.T) {
	input, err := NewTensor([]byte{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	output, err := Transpose(input)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(input)
	fmt.Println(output)
}

func TestTensor_MarshalJSON(t *testing.T) {
	input, err := NewTensor([]byte{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	jb, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}

	output := &Tensor[byte]{}
	if err := json.Unmarshal(jb, output); err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(input.value, output.value) {
		t.Fatal("input value is not equal to output value")
	}

	if len(input.shape) != len(output.shape) {
		t.Fatal("input shape is not same length as output shape")
	}

	for i := range input.shape {
		if input.shape[i] != output.shape[i] {
			t.Fatal("input shape value is not same as output shape value")
		}
	}
}

func TestTensor_MarshalJSONComplex128(t *testing.T) {
	input, err := NewTensor(
		[]complex128{
			complex(float64(1.2), float64(1.3)),
			complex(float64(1.3), float64(1.4)),
			complex(float64(1.4), float64(1.5)),
			complex(float64(1.5), float64(1.6)),
			complex(float64(1.6), float64(1.7)),
			complex(float64(1.7), float64(1.8)),
		}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	jb, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(string(jb))

	output := &Tensor[complex128]{}
	if err := json.Unmarshal(jb, output); err != nil {
		t.Fatal(err)
	}

	if len(input.shape) != len(output.shape) {
		t.Fatal("input shape is not same length as output shape")
	}

	for i := range input.value {
		if input.value[i] != output.value[i] {
			t.Fatal("input and output values do not match")
		}
	}

	for i := range input.shape {
		if input.shape[i] != output.shape[i] {
			t.Fatal("input shape value is not same as output shape value")
		}
	}
}

func TestTensor_MarshalJSONComplex64(t *testing.T) {
	input, err := NewTensor(
		[]complex64{
			complex(float32(1.2), float32(1.3)),
			complex(float32(1.3), float32(1.4)),
			complex(float32(1.4), float32(1.5)),
			complex(float32(1.5), float32(1.6)),
			complex(float32(1.6), float32(1.7)),
			complex(float32(1.7), float32(1.8)),
		}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	jb, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(string(jb))

	output := &Tensor[complex64]{}
	if err := json.Unmarshal(jb, output); err != nil {
		t.Fatal(err)
	}

	if len(input.shape) != len(output.shape) {
		t.Fatal("input shape is not same length as output shape")
	}

	for i := range input.value {
		if input.value[i] != output.value[i] {
			t.Fatal("input and output values do not match")
		}
	}

	for i := range input.shape {
		if input.shape[i] != output.shape[i] {
			t.Fatal("input shape value is not same as output shape value")
		}
	}
}

func TestNewFromFunc(t *testing.T) {
	f := func(int) float64 {
		return rand.Float64()
	}

	tensor, err := NewFromFunc(f, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(tensor)
}

func TestZipToComplex128(t *testing.T) {
	f := func(int) float64 {
		return rand.Float64()
	}

	realT, err := NewFromFunc(f, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	imagT, err := NewFromFunc(f, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	tensor, err := Complex128(realT, imagT)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(tensor)
}

func TestZipToComplex64(t *testing.T) {
	f := func(int) float32 {
		return rand.Float32()
	}

	realT, err := NewFromFunc(f, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	imagT, err := NewFromFunc(f, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	tensor, err := Complex64(realT, imagT)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(tensor)
}

func TestNewFromFuncComplex128(t *testing.T) {
	f := func(int) complex128 {
		return complex(rand.Float64(), rand.Float64())
	}

	tensor, err := NewFromFunc(f, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(tensor)

	invT, err := MatrixInverse(tensor)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(invT)
}

func TestTensor_Apply(t *testing.T) {
	tensor, err := NewTensor([]string{"a", "b", "c", "d"}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	tensor.Apply(
		func(input string) string {
			return strings.ToUpper(input)
		},
	)

	fmt.Println(tensor)
}
