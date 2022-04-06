package tfutil

import (
	"testing"
)

func TestTensor_ReshapeInt32(t *testing.T) {
	m, err := NewTensor([]int32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	value := clone(m.value)
	shape := clone(m.shape)

	if err := m.Reshape(3, 2); err != nil {
		t.Fatal(err)
	}

	if !equal(value, m.Value()) {
		t.Fatal("values are not equal")
	}

	if !equal(shape, []int{2, 3}) {
		t.Fatal("original shape not equal to expected")
	}

	if !equal(m.shape, []int{3, 2}) {
		t.Fatal("final shape not equal to expected")
	}
}

func TestTensor_ReshapeString(t *testing.T) {
	m, err := NewTensor([]string{"abcd", "1234", "c", "d0", "ee", "ffff"}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	value := clone(m.value)
	shape := clone(m.shape)

	if err := m.Reshape(3, 2); err != nil {
		t.Fatal(err)
	}

	if !equal(value, m.Value()) {
		t.Fatal("values are not equal")
	}

	if !equal(shape, []int{2, 3}) {
		t.Fatal("original shape not equal to expected")
	}

	if !equal(m.shape, []int{3, 2}) {
		t.Fatal("final shape not equal to expected")
	}
}

func TestTensor_ReshapeBool(t *testing.T) {
	m, err := NewTensor([]bool{true, true, false, true, false, false}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	value := clone(m.value)
	shape := clone(m.shape)

	if err := m.Reshape(3, 2); err != nil {
		t.Fatal(err)
	}

	if !equal(value, m.Value()) {
		t.Fatal("values are not equal")
	}

	if !equal(shape, []int{2, 3}) {
		t.Fatal("original shape not equal to expected")
	}

	if !equal(m.shape, []int{3, 2}) {
		t.Fatal("final shape not equal to expected")
	}
}
