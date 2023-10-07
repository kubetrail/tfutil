# tfutil
`tfutil` is a generics based Go library on top of 
[TensorFlow](https://github.com/tensorflow/tensorflow) 
[Go interface](https://github.com/wamuir/graft) to make it
easier to work with tensors. In particular, this package
intends to enable easier use of operators on tensors and
serialization of data to-from tensor objects.

> TF version v2.14.0

# disclaimer
> The use of this library does not guarantee security or usability for any
> particular purpose. Please review the code and use at your own risk.

> Please also note that the API is not yet stable and can change.

## installation
This step assumes you have [Go compiler toolchain](https://go.dev/dl/)
installed on your system with version at least Go 1.21.1. The library
makes use of `go` generics.

First make sure you have downloaded and installed `TensorFlow`
[C-library](https://www.tensorflow.org/install/lang_c) and make 
sure you are able to build and run the "hello-world" as 
described on that page.

> Please use TF version v2.14.0. Older versions may not work

Download this repo to a folder and cd to it.
```bash
go test ./...
```

## usage
A type parameterized `Tensor` can be instantiated using one of these following
ways:
```go
func NewTensor[T PrimitiveTypes](value []T, shape ...int) (*Tensor[T], error) {...}
func NewTensorFromFunc[T PrimitiveTypes](f func(int) T, shape ...int) (*Tensor[T], error) {...}
```

Another way to create a tensor is to input any value such as a slice of slice,
however, type parameter needs to be explicitly provided
```go
func NewTensorFromAny[T PrimitiveTypes](value any) (*Tensor[T], error) {...}
```

where, `PrimitiveTypes` are:
```go
type PrimitiveTypes interface {
	bool |
		int8 | int16 | int32 | int64 |
		uint8 | uint16 | uint32 | uint64 |
		float32 | float64 |
		complex64 | complex128 |
		string
}
```

For instance a random `5x5` matrix of data type `float64` can be created as follows:
```go
matrix, err := tfutil.NewFromFunc(
		func(int) float64 { return rand.Float64() }, // generator function
		5, // rows
		5, // columns
		)
```

Similarly, a random `5x5` matrix of data type `int64` can be created as follows:
```go
matrixInt64, err := NewTensorFromFunc(
        func(int) int64 { return int64(rand.Intn(100)) }, // generator function
        5, // rows
        5, // columns
        )
```

Matrix can be printed as follows:
```go
fmt.Println(matrixInt64)
```
```bash
[ # matrix shape: [5 5], dataType: int64
      3   20   71   25   90     
      3   34   48   14   71     
     54   19   48   94   92     
     22   75   69   50   54     
     52    1   39    2    1     
]
```

Use `Cast` to cast the matrix data type to a different data type.
For instance, passing a type parameter `float64` to the function
will cast input matrix, which is of data type `int64` to `float64`
```go
matrixF64, err := Cast[float64](matrixInt64)
```

### operators
Operations such as matrix inversion can be performed on tensors as follows.
```go
err := matrixF64.Apply(MatrixInverseOp)
```

> Please note that it is not possible to perform all checks at the compile
> time. So please apply operators with the knowledge of what they do. For
> instance, applying MatrixInverseOp on a tensor of data type int64 will
> cause runtime failures

### serialization
```go
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
	
	fmt.Println(string(jb))
```
This results in JSON serialization as follows:
```json
{"type":"tensor","tfDataType":"Uint8","goDataType":"uint8","shape":[2,3],"value":"AQIDBAUG"}
```

The serialized form can be parsed back into a tensor parametrized by the corresponding data type.

### json serialization of complex data
`complex64` and `complex128` data types are not supported natively by JSON serialization.
These values, therefore, are separated out into real and imaginary parts as follows
```go
    input, _ := NewFromFunc(
		func(int) complex128 { return complex(rand.Float64(), rand.Float64()) }, 2, 3)

	b, _ := json.Marshal(input)
	fmt.Println(string(b))

	output := &Tensor[complex128]{}
	_ = json.Unmarshal(b, output)

	b, _ = output.PrettyPrint()
	fmt.Println(string(b))
```

The JSON serialization looks as follows:
```json
{
  "type": "tensor",
  "tfDataType": "Complex128",
  "goDataType": "complex128",
  "shape": [
    2,
    3
  ],
  "real64": [
    0.6046602879796196,
    0.6645600532184904,
    0.4246374970712657,
    0.06563701921747622,
    0.09696951891448456,
    0.5152126285020654
  ],
  "imag64": [
    0.9405090880450124,
    0.4377141871869802,
    0.6868230728671094,
    0.15651925473279124,
    0.30091186058528707,
    0.8136399609900968
  ]
}
```
