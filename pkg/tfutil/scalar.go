package tfutil

import (
	"encoding/json"
	"fmt"

	tf "github.com/wamuir/graft/tensorflow"
)

// Scalar represents a singular value type parametrized by supported types
type Scalar[T PrimitiveTypes] struct {
	value T
}

// scalarSerializer serializes a scalar to/from JSON data
type scalarSerializer[T PrimitiveTypes] struct {
	Type       string   `json:"type,omitempty"`
	TfDataType string   `json:"tfDataType,omitempty"`
	GoDataType string   `json:"goDataType,omitempty"`
	Value      *T       `json:"value,omitempty"`
	Real64     *float64 `json:"real64,omitempty"` // for complex128 data type
	Imag64     *float64 `json:"imag64,omitempty"` // imaginary part
	Real32     *float32 `json:"real32,omitempty"` // for complex64 data type
	Imag32     *float32 `json:"imag32,omitempty"` // imaginary part
}

// NewScalar generates a new scalar parametrized by a data type
func NewScalar[T PrimitiveTypes](value T) *Scalar[T] {
	return &Scalar[T]{value: value}
}

// Value retrieves underlying value of the scalar
func (g *Scalar[T]) Value() T {
	return g.value
}

// String prints numpy representation of scalar value
func (g *Scalar[T]) String() string {
	return fmt.Sprint(g.value)
}

// MarshalJSON serializes scalar with additional metadata such
// as data types in tensorflow and go. Use scalar in
// json.Marshal for this method to be called indirectly.
func (g *Scalar[T]) MarshalJSON() ([]byte, error) {
	tfTensor, err := tf.NewTensor(g.value)
	if err != nil {
		return nil, fmt.Errorf("failed to get tf tensor: %w", err)
	}

	switch any(*new(T)).(type) {
	case complex128:
		c := any(g.value).(complex128)
		real64, imag64 := real(c), imag(c)
		return json.Marshal(
			scalarSerializer[T]{
				Type:       TypeScalar,
				TfDataType: dataTypeMap[tfTensor.DataType()],
				GoDataType: fmt.Sprintf("%T", tfTensor.Value()),
				Real64:     &real64,
				Imag64:     &imag64,
			},
		)
	case complex64:
		c := any(g.value).(complex64)
		real32, imag32 := real(c), imag(c)
		return json.Marshal(
			scalarSerializer[T]{
				Type:       TypeScalar,
				TfDataType: dataTypeMap[tfTensor.DataType()],
				GoDataType: fmt.Sprintf("%T", tfTensor.Value()),
				Real32:     &real32,
				Imag32:     &imag32,
			},
		)
	default:
		return json.Marshal(
			scalarSerializer[T]{
				Type:       TypeScalar,
				TfDataType: dataTypeMap[tfTensor.DataType()],
				GoDataType: fmt.Sprintf("%T", tfTensor.Value()),
				Value:      &g.value,
			},
		)
	}
}

// UnmarshalJSON parses serialized scalar value
func (g *Scalar[T]) UnmarshalJSON(data []byte) error {
	s := &scalarSerializer[T]{}
	if err := json.Unmarshal(data, s); err != nil {
		return fmt.Errorf("failed to parse input: %w", err)
	}

	if s.Type != TypeScalar {
		return fmt.Errorf("type is not %s", TypeScalar)
	}

	zeroValue := *new(T)
	if s.GoDataType != fmt.Sprintf("%T", zeroValue) {
		return fmt.Errorf("expected go data type to be %T, received %s in serialized json", zeroValue, s.GoDataType)
	}

	// if complex data is received, separate it out into real and imaginary parts
	switch any(*new(T)).(type) {
	case complex128:
		if s.Real64 == nil || s.Imag64 == nil {
			return fmt.Errorf("did not receive expected data in real64 and/or imag64 fields")
		}

		if s.Real32 != nil || s.Imag32 != nil || s.Value != nil {
			return fmt.Errorf("data values found in at least one of real32, imag32 or value fields, not expected to be there")
		}

		g.value = any(complex(*s.Real64, *s.Imag64)).(T)
	case complex64:
		if s.Real32 == nil || s.Imag32 == nil {
			return fmt.Errorf("did not receive expected data in real32 and/or imag32 fields")
		}

		if s.Real64 != nil || s.Imag64 != nil || s.Value != nil {
			return fmt.Errorf("data values found in at least one of real64, imag64 or value fields, not expected to be there")
		}

		g.value = any(complex(*s.Real32, *s.Imag32)).(T)
	default:
		if s.Value == nil {
			return fmt.Errorf("data not found in value field")
		}
		g.value = *s.Value
	}

	return nil
}

// Marshal produces an instance of upstream tensor based on scalar value
func (g *Scalar[T]) Marshal() (*tf.Tensor, error) {
	tfTensor, err := tf.NewTensor(g.value)
	if err != nil {
		return nil, err
	}

	return tfTensor, nil
}

// Unmarshal populates receiver scalar using value from input upstream tensor
func (g *Scalar[T]) Unmarshal(tfTensor *tf.Tensor) error {
	value, ok := tfTensor.Value().(T)
	if !ok {
		return fmt.Errorf("type assertion failed, expected %T, received %T", g.value, tfTensor.Value())
	}

	g.value = value
	return nil
}

// Clone creates a clone of receiver scalar
func (g *Scalar[T]) Clone() *Scalar[T] {
	return NewScalar(g.value)
}
