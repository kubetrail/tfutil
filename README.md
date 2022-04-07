# tfutil
`tfutil` is a Go library on top of `TensorFlow` Go interface to make it
easier to work with tensors. As the name suggests, this library aims to
provide utility functions for tensors such as serialization and type safety
using type parametrization

# disclaimer
> The use of this library does not guarantee security or usability for any
> particular purpose. Please review the code and use at your own risk.

## installation
This step assumes you have [Go compiler toolchain](https://go.dev/dl/)
installed on your system with version at least Go 1.18. The library
makes use of `go` generics.

First make sure you have downloaded and installed `TensorFlow`
[C-library](https://www.tensorflow.org/install/lang_c) and make 
sure you are able to build and run the "hello-world" as 
described on that page.

Furthermore, the upstream [tensorflow](https://github.com/tensorflow/tensorflow) 
code does not provide protocol buffer files for Go and does not currently
have a `go.mod` file, which makes it harder to use. To ease the workflow,
tensorflow code has been [forked](https://github.com/kubetrail-labs/tensorflow)
and needs to be added via a `replace` clause in your `go.mod` file. See 
usage below.

Download this repo to a folder and cd to it.
```bash
go test -v ./...
```

## usage
### example matrix inversion
Create a new go module and `go get` this repo:
```bash
go get github.com/kubetrail/tfutil
```

Pl. make sure to have following `replace` clause to pin to a 
particular commit ID's of the forked tensorflow repo:
```
replace github.com/tensorflow/tensorflow/tensorflow/go => github.com/kubetrail-labs/tensorflow/tensorflow/go v0.0.0-20220330185145-9a3cb0962c98
```

Write `Go` code:
```go
package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/kubetrail/tfutil/pkg/tfutil"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	// create a random square matrix using a generator func
	x, err := tfutil.NewFromFunc(
		func(int) float64 { return rand.Float64() }, 5, 5)
	if err != nil {
		log.Fatal(err)
	}

	y, err := tfutil.MatrixInverse(x)
	if err != nil {
		log.Fatal(err)
	}

	z, err := tfutil.MatrixMultiply(x, y)
	if err != nil {
		log.Fatal(err)
	}

	// apply transformation operators in sequence with
	// last one being applied first. In this case
	// the output will be from abs(round(z))
	z, err = tfutil.ApplyOperators(z, op.Abs, op.Round)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("matrix:")
	fmt.Println(x)

	fmt.Println("matrix inverse:")
	fmt.Println(y)

	fmt.Println("matrix multiplied by its inverse is identity matrix")
	fmt.Println(z)
}
```

Running this code produces:
```text
└─ $ ▶ go run main.go 
matrix:
[ # matrix shape: [5 5], dataType: float64
      0.6046602879796196    0.9405090880450124    0.6645600532184904    0.4377141871869802    0.4246374970712657     
      0.6868230728671094   0.06563701921747622   0.15651925473279124   0.09696951891448456   0.30091186058528707     
      0.5152126285020654    0.8136399609900968   0.21426387258237492     0.380657189299686   0.31805817433032985     
      0.4688898449024232   0.28303415118044517   0.29310185733681576    0.6790846759202163   0.21855305259276428     
     0.20318687664732285     0.360871416856906    0.5706732760710226    0.8624914374478864   0.29311424455385804     
]

matrix inverse:
[ # matrix shape: [5 5], dataType: float64
       1.326302317049744   -0.13382937179671123   -1.5241040918148205   3.4709920537115804   -2.7182882092548732     
      0.5647043565175299     -1.211411206833943    0.9216998312774841   0.4173078118858744    -0.885745783269239     
      2.9600099759890797    -0.5993164120459926   -3.4009914688680962   1.1673572264105623   -0.8529307712748238     
     -1.0256616804088645    -0.5547649814313621      0.57114234242179   1.3082838291342402    0.4601750301751597     
      -4.359552916330332      4.383444275003038    4.8626513930315705   -9.042268391824125     6.692982813192693     
]

matrix multiplied by its inverse is identity matrix
[ # matrix shape: [5 5], dataType: float64
     1   0   0   0   0     
     0   1   0   0   0     
     0   0   1   0   0     
     0   0   0   1   0     
     0   0   0   0   1     
]
```

Verify matrix inversion using `Julia`:
```text
└─ $ ▶ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.7.2 (2022-02-06)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> x = [ # matrix shape: [5 5], dataType: float64
              0.6046602879796196      0.9405090880450124      0.6645600532184904      0.4377141871869802      0.4246374970712657      
              0.6868230728671094     0.06563701921747622     0.15651925473279124     0.09696951891448456     0.30091186058528707      
              0.5152126285020654      0.8136399609900968     0.21426387258237492       0.380657189299686     0.31805817433032985      
              0.4688898449024232     0.28303415118044517     0.29310185733681576      0.6790846759202163     0.21855305259276428      
             0.20318687664732285       0.360871416856906      0.5706732760710226      0.8624914374478864     0.29311424455385804      
       ]
5×5 Matrix{Float64}:
 0.60466   0.940509  0.66456   0.437714   0.424637
 0.686823  0.065637  0.156519  0.0969695  0.300912
 0.515213  0.81364   0.214264  0.380657   0.318058
 0.46889   0.283034  0.293102  0.679085   0.218553
 0.203187  0.360871  0.570673  0.862491   0.293114

julia> inv(x)
5×5 Matrix{Float64}:
  1.3263    -0.133829  -1.5241     3.47099   -2.71829
  0.564704  -1.21141    0.9217     0.417308  -0.885746
  2.96001   -0.599316  -3.40099    1.16736   -0.852931
 -1.02566   -0.554765   0.571142   1.30828    0.460175
 -4.35955    4.38344    4.86265   -9.04227    6.69298

julia> 
```

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

## shipping binary
Unlike pure Go binaries, the use of `libtensorflow.so` as a dynamic dependency
puts restrictions on where the compiled binary can run. The final deployment
package consists of `libtensorflow*.so` files, compiled Go binary and
environment variable `LD_LIBRARY_PATH`. Full description can be found
in the [Dockerfile](cmd/matrix-inverse/Dockerfile)

Build it as follows:
```bash
cd example/matrix-inverse
docker build -t tf-example ./
```

Run:
```bash
docker run --rm tf-example matrix-inverse
input matrix: [[0.6046602879796196 0.9405090880450124 0.6645600532184904 0.4377141871869802 0.4246374970712657] [0.6868230728671094 0.06563701921747622 0.15651925473279124 0.09696951891448456 0.30091186058528707] [0.5152126285020654 0.8136399609900968 0.21426387258237492 0.380657189299686 0.31805817433032985] [0.4688898449024232 0.28303415118044517 0.29310185733681576 0.6790846759202163 0.21855305259276428] [0.20318687664732285 0.360871416856906 0.5706732760710226 0.8624914374478864 0.29311424455385804]]
inverted matrix: [[1.326302317049744 -0.13382937179671123 -1.5241040918148205 3.4709920537115804 -2.7182882092548732] [0.5647043565175299 -1.211411206833943 0.9216998312774841 0.4173078118858744 -0.885745783269239] [2.9600099759890797 -0.5993164120459926 -3.4009914688680962 1.1673572264105623 -0.8529307712748238] [-1.0256616804088645 -0.5547649814313621 0.57114234242179 1.3082838291342402 0.4601750301751597] [-4.359552916330332 4.383444275003038 4.8626513930315705 -9.042268391824125 6.692982813192693]]
```

In case of issues you can inspect dependency chain as follows. Make sure all required
files are reachable and there are no "not found" errors
```bash
docker run --rm -it tf-example bash
root@c549661fd90f:/# ldd /tf/matrix-inverse 
	linux-vdso.so.1 (0x00007ffeca553000)
	libtensorflow.so.2 => /usr/local/lib/libtensorflow.so.2 (0x00007fde14ad4000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fde14aaf000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fde148bd000)
	libtensorflow_framework.so.2 => /usr/local/lib/libtensorflow_framework.so.2 (0x00007fde12b61000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fde12b5b000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fde12a0c000)
	librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007fde12a00000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fde1281e000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fde12803000)
	/lib64/ld-linux-x86-64.so.2 (0x00007fde235f6000)
root@c549661fd90f:/# 
```

## changes to upstream tensorflow code
Following changes were made to generate protocol buffer Go files for upstream version `v2.8.0`:
* A `go.mod` file was added
* Assumptions on folders for writing new protobuf go files were changed in `generate.sh`. This
allowed running `go generate` from the downloaded folder

More details at the 
[commit history](https://github.com/kubetrail-labs/tensorflow/commit/9a3cb0962c983435b9d103fe9f8e2ee9fe0cb000)

## building libtensorflow
`libtensorflow` is not available as a [precompiled c-library](https://www.tensorflow.org/install/lang_c) 
from upstream for `arm64` arch. Below are unofficial steps to build it for `arm64`:

First setup a Raspberry Pi 4 with 8GB memory 
a [64-bit OS](https://downloads.raspberrypi.org/raspios_arm64/images)

Setup 8GB swap
```bash
sudo swapoff -a
sudo dd if=/dev/zero of=/swapfile bs=1G count=8
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

Add following line to `/etc/fstab`:
```
/swapfile swap swap defaults 0 0
```

Download [bazel](https://github.com/bazelbuild/bazel/releases/download/4.2.1/bazel-4.2.1-linux-arm64)

Install `GCC` and build tools:
```bash
sudo apt install build-essential
```

Install `java`:
```bash
sudo apt install default-jdk
```

Install `pip`:
```bash
sudo apt install python3-dev python3-venv python3-pip
pip install -U --user pip numpy wheel packaging
pip install -U --user keras_preprocessing --no-deps
```

Clone `tensorflow` at version `v2.8.0`:
```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout tags/v2.8.0
```

reboot now

Configure tensorflow build
```bash
./configure # accept all default answers
```

Build tensorflow library and test... be patient, this can take several hours.
```bash
nohup bazel test --config opt //tensorflow/tools/lib_package:libtensorflow_test &
```

reboot now

```bash
nohup bazel build --config opt //tensorflow/tools/lib_package:libtensorflow &
```

## references
* https://pgaleone.eu/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/
