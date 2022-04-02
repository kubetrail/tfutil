package tfutil

import (
	_ "embed"
	"encoding/json"
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// load models that provide helper functionality.
// reshape-string-tensor.pb is an exported model that can
// reshape a string tensor.
// an exported graph was needed since reshape of string
// tensor is not natively supported by the go code

//go:embed models/proto/reshape-string-tensor.pb
var graphReshapeStringTensor []byte

const (
	TypeScalar = "scalar"
	TypeTensor = "tensor"
)

// dataTypeMap stores named representation of tf data types
var dataTypeMap = map[tf.DataType]string{
	tf.Float:      "Float",
	tf.Double:     "Double",
	tf.Int32:      "Int32",
	tf.Uint32:     "Uint32",
	tf.Uint8:      "Uint8",
	tf.Int16:      "Int16",
	tf.Int8:       "Int8",
	tf.String:     "String",
	tf.Complex64:  "Complex64", //tf.Complex: "Complex", // duplicate key 8
	tf.Int64:      "Int64",
	tf.Uint64:     "Uint64",
	tf.Bool:       "Bool",
	tf.Qint8:      "Qint8",
	tf.Quint8:     "Quint8",
	tf.Qint32:     "Qint32",
	tf.Bfloat16:   "Bfloat16",
	tf.Qint16:     "Qint16",
	tf.Quint16:    "Quint16",
	tf.Uint16:     "Uint16",
	tf.Complex128: "Complex128",
	tf.Half:       "Half",
}

// PrimitiveTypes are type constraints for supported underlying types
type PrimitiveTypes interface {
	bool |
		int8 | int16 | int32 | int64 |
		uint8 | uint16 | uint32 | uint64 |
		float32 | float64 |
		complex64 | complex128 |
		string
}

// Scalar represents a singular value type parametrized by supported types
type Scalar[T PrimitiveTypes] struct {
	value T
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
	return json.Marshal(
		struct {
			Type       string `json:"type,omitempty"`
			TfDataType string `json:"tfDataType,omitempty"`
			GoDataType string `json:"goDataType,omitempty"`
			Shape      []int  `json:"shape"`
			Value      T      `json:"value"`
		}{
			Type:       TypeScalar,
			TfDataType: dataTypeMap[tfTensor.DataType()],
			GoDataType: fmt.Sprintf("%T", tfTensor.Value()),
			Shape:      []int{},
			Value:      g.value,
		},
	)
}

// Tensor is a generic non-scalar data structures that includes
// vectors, matrices and higher dimensional structures.
// Tensor's representation is a slice because it is easier
// to work with compared to multidimensional slices.
// Shape stores the underlying dimensionality.
type Tensor[T PrimitiveTypes] struct {
	value []T
	shape []int
}

// NewTensor creates a new tensor with specified dimensions. If no dimension
// argument is specified, it is assumed that a vector is being created
// and the shape assumes value equal to the length of the input slice
func NewTensor[T PrimitiveTypes](value []T, shape ...int) (*Tensor[T], error) {
	if len(shape) == 0 {
		shape = []int{len(value)}
	}

	n, err := numElements(shape)
	if err != nil {
		return nil, fmt.Errorf("invalid shape: %w", err)
	}

	if len(value) != n {
		return nil, fmt.Errorf("mismatch between length of value and shape")
	}

	return &Tensor[T]{
		value: value,
		shape: shape,
	}, nil
}

// String prints a numpy version of tensor value
// ignoring any errors that may occur during data
// transformation to numpy version
func (g *Tensor[T]) String() string {
	m, _ := g.GetMultiDimSlice()
	return fmt.Sprint(m)
}

// MarshalJSON serializes tensor with additional metadata such
// as tensorflow data type and go data type. Use tensor
// in json.Marshal for this method to be called indirectly.
func (g *Tensor[T]) MarshalJSON() ([]byte, error) {
	tfTensor, err := g.Marshal()
	if err != nil {
		return nil, fmt.Errorf("failed to get tf tensor: %w", err)
	}
	return json.Marshal(
		struct {
			Type       string `json:"type,omitempty"`
			TfDataType string `json:"tfDataType,omitempty"`
			GoDataType string `json:"goDataType,omitempty"`
			Shape      []int  `json:"shape"`
			Value      []T    `json:"value"`
		}{
			Type:       TypeTensor,
			TfDataType: dataTypeMap[tfTensor.DataType()],
			GoDataType: fmt.Sprintf("%T", tfTensor.Value()),
			Shape:      g.shape,
			Value:      g.value,
		},
	)
}

// Value returns underlying slice representation of the tensor
func (g *Tensor[T]) Value() []T {
	return g.value
}

// Shape returns tensor shape
func (g *Tensor[T]) Shape() []int {
	return g.shape
}

// NumElements is the total number of elements in the tensor
func (g *Tensor[T]) NumElements() int {
	return len(g.value)
}

func (g *Tensor[T]) SetElement(value T, indices ...int) error {
	index, err := g.indicesToIndex(indices)
	if err != nil {
		return err
	}

	g.value[index] = value
	return nil
}

// GetElement retrieves an element indexed by indices. This is
// a slow method, for faster access it is recommended to obtain
// a multidimensional slice and index off of that.
func (g *Tensor[T]) GetElement(indices ...int) (T, error) {
	zt := *new(T)

	index, err := g.indicesToIndex(indices)
	if err != nil {
		return zt, err
	}

	return g.value[index], nil
}

// GetMultiDimSlice fetches a multi-dimensional slice corresponding
// to underlying slice. Please note that it is users responsibility
// to perform type assertion correctly on returned value
func (g *Tensor[T]) GetMultiDimSlice() (any, error) {
	tfTensor, err := g.Marshal()
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tfTensor: %w", err)
	}

	return tfTensor.Value(), nil
}

// Marshal returns an instance of upstream tensorflow tensor object.
// string tensor reshaping is currently not supported natively in go.
// it is, however, possible to reshape it via a tf session running over
// a graphdef that was generated using python code for reshape function
func (g *Tensor[T]) Marshal() (*tf.Tensor, error) {
	tfTensor, err := tf.NewTensor(g.value)
	if err != nil {
		return nil, fmt.Errorf("failed to create a tensor: %w", err)
	}

	// if the receiver is a vector, there is no need to reshape
	if len(g.shape) == 1 {
		return tfTensor, nil
	}

	// create a new variable, wrap it in empty interface then
	// do type switch on it so selective treated can be done for
	// string tensors which have some exceptions and unsupported
	// features in go interface
	switch any(*new(T)).(type) {
	case string:
		// string tensor reshape is currently not supported in go interface.
		// below is a workaround via graph-def generated using a python model.
		// see models/proto/reshape-string-tensor.py for more info

		// import the graph
		graph := tf.NewGraph()
		if err := graph.Import(graphReshapeStringTensor, ""); err != nil {
			return nil, fmt.Errorf("failed to import graph: %w", err)
		}

		// prepare shape of the matrix
		// ensure shape dimension is passed as int32 because python model
		// that was used to generate protobuf expects it to be int32 and
		// errors out otherwise
		shape := make([]int32, len(g.shape))
		for i, v := range g.shape {
			shape[i] = int32(v)
		}
		dim, err := tf.NewTensor(shape)
		if err != nil {
			return nil, fmt.Errorf("failed to create shape tensor: %w", err)
		}

		// operation names can be found via following code snippet
		/*// print available operations in the graph
		for i, operation := range graph.Operations() {
			fmt.Println(">>>", i, operation.Name())
		}*/

		// prepare data feed specifying names of the operation.
		// names x and dim come from python code, see def of reshape
		// function taking inputs x and dim
		feeds := map[tf.Output]*tf.Tensor{
			graph.Operation("x").Output(0):   tfTensor,
			graph.Operation("dim").Output(0): dim,
		}

		// prepare data outputs from tensorflow run.
		// Identity is the final output point of the graph.
		fetches := []tf.Output{
			graph.Operation("Identity").Output(0),
		}

		// start new session
		sess, err := tf.NewSession(
			graph,
			&tf.SessionOptions{},
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create new tf session: %w", err)
		}
		defer sess.Close()

		// run session feeding feeds and fetching fetches
		out, err := sess.Run(feeds, fetches, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to run tf session: %w", err)
		}

		if len(out) != 1 {
			return nil, fmt.Errorf("string reshape tf session output generated output length is not equal to 1")
		}

		return out[0], nil
	default:
		shape := make([]int64, len(g.shape))
		for i := range shape {
			shape[i] = int64(g.shape[i])
		}
		if err := tfTensor.Reshape(shape); err != nil {
			return nil, fmt.Errorf("")
		}

		return tfTensor, nil
	}
}

// Unmarshal populates receiver based on input upstream tensor object.
// string tensor reshaping is currently not supported natively in go.
// it is, however, possible to reshape it via a tf session running over
// a graphdef that was generated using python code for reshape function
func (g *Tensor[T]) Unmarshal(tfTensor *tf.Tensor) error {
	tfShape := tfTensor.Shape()
	shape := make([]int, len(tfShape))
	for i := range shape {
		shape[i] = int(tfShape[i])
	}

	// if the shape is that of a vector, then there is no need to reshape
	if len(shape) == 1 {
		values, ok := tfTensor.Value().([]T)
		if !ok {
			return fmt.Errorf("type assertion failed, expected %T, received %T", g.value, tfTensor.Value())
		}

		g.value = values
		g.shape = shape

		return nil
	}

	n, err := numElements(shape)
	if err != nil {
		return fmt.Errorf("invalid input tensor shape: %w", err)
	}

	// create a new variable, wrap it in empty interface then
	// do type switch on it so selective treated can be done for
	// string tensors which have some exceptions and unsupported
	// features in go interface
	switch any(*new(T)).(type) {
	case string:
		// string tensor reshape is currently not supported in go interface.
		// below is a workaround via graph-def generated using a python model.
		// see models/proto/reshape-string-tensor.py for more info

		// import the graph
		graph := tf.NewGraph()
		if err := graph.Import(graphReshapeStringTensor, ""); err != nil {
			return fmt.Errorf("failed to import graph: %w", err)
		}

		// prepare shape of the matrix
		// ensure shape dimension is passed as int32 because python model
		// that was used to generate protobuf expects it to be int32 and
		// errors out otherwise
		dim, err := tf.NewTensor([]int32{int32(n)})
		if err != nil {
			return fmt.Errorf("failed to create shape tensor: %w", err)
		}

		// operation names can be found via following code snippet
		/*// print available operations in the graph
		for i, operation := range graph.Operations() {
			fmt.Println(">>>", i, operation.Name())
		}*/

		// prepare data feed specifying names of the operation.
		// names x and dim come from python code, see def of reshape
		// function taking inputs x and dim
		feeds := map[tf.Output]*tf.Tensor{
			graph.Operation("x").Output(0):   tfTensor,
			graph.Operation("dim").Output(0): dim,
		}

		// prepare data outputs from tensorflow run.
		// Identity is the final output point of the graph.
		fetches := []tf.Output{
			graph.Operation("Identity").Output(0),
		}

		// start new session
		sess, err := tf.NewSession(
			graph,
			&tf.SessionOptions{},
		)
		if err != nil {
			return fmt.Errorf("failed to create new tf session: %w", err)
		}
		defer sess.Close()

		// run session feeding feeds and fetching fetches
		out, err := sess.Run(feeds, fetches, nil)
		if err != nil {
			return fmt.Errorf("failed to run tf session: %w", err)
		}

		if len(out) != 1 {
			return fmt.Errorf("string reshape tf session output generated output length is not equal to 1")
		}

		// reshape output data as vector
		values, ok := out[0].Value().([]T)
		if !ok {
			return fmt.Errorf("output type from tf session run is not []string")
		}

		g.value = values
		g.shape = shape

		return nil
	default:
		if err := tfTensor.Reshape([]int64{int64(n)}); err != nil {
			return fmt.Errorf("failed to reshape input tensor: %w", err)
		}

		value, ok := tfTensor.Value().([]T)
		if !ok {
			return fmt.Errorf("type assertion failed, expected %T, received %T", g.value, tfTensor.Value())
		}

		g.value = value
		g.shape = shape

		if err := tfTensor.Reshape(tfShape); err != nil {
			return fmt.Errorf("failed to reshape input tensor back to original shape: %w", err)
		}

		return nil
	}
}

// indicesToIndex converts the dimensional indices (or subscripts)
// to a positional index in the slice... all tensors are represented
// as []T, so a positional index is simply an index on that slice
func (g *Tensor[T]) indicesToIndex(indices []int) (int, error) {
	if len(indices) != len(g.shape) {
		return 0, fmt.Errorf(
			"invalid number of indices, expected %d, got %d", len(g.shape), len(indices),
		)
	}

	shape := g.shape

	// weights apply to each index. lower the index,
	// higher is its weight, in the sense of how many
	// elements it encapsulates in the tensor
	weights := make([]int, len(shape))
	for i := len(weights) - 1; i >= 0; i-- {
		weights[i] = 1
		if i < len(weights)-1 {
			weights[i] = shape[i+1] * weights[i+1]
		}
	}

	index := 0
	for i, v := range indices {
		if v >= shape[i] {
			return 0, fmt.Errorf(
				"index %d is %d and needs to be less than %d", i, v, shape[i],
			)
		}
		index += v * weights[i]
	}

	return index, nil
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

// Clone creates a clone of receiver tensor
func (g *Tensor[T]) Clone() (*Tensor[T], error) {
	value := make([]T, len(g.value))
	shape := make([]int, len(g.shape))

	for i, v := range g.value {
		value[i] = v
	}

	for i, v := range g.shape {
		shape[i] = v
	}

	out, err := NewTensor(value, shape...)
	if err != nil {
		return nil, err
	}

	return out, nil
}

// numElements is the total number of elements represented by
// shape slice.
// error is returned if shape value is negative or zero.
func numElements(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, nil
	}

	n := shape[0]
	if n <= 0 {
		return -1, fmt.Errorf("please provide positive shape values")
	}

	if len(shape) == 1 {
		return n, nil
	}

	for i := 1; i < len(shape); i++ {
		if shape[i] <= 0 {
			return -1, fmt.Errorf("please provide positive shape values")
		}
		n *= shape[i]
	}

	return n, nil
}
