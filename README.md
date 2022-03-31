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
go test -v ./
```

## usage
### example matrix inversion
In order to use [tf.linalg.inv](https://www.tensorflow.org/api_docs/python/tf/linalg/inv) API
via Go, we need to first invoke it via Python using concrete data types and export the model
as a protocol buffer graph. Once this graph is exported, Python is no longer required in the
workflow.

Python code to generate matrix inversion [graph](https://www.tensorflow.org/guide/intro_to_graphs):
```python
import tensorflow as tf

f = tf.function(tf.linalg.inv)
g = f.get_concrete_function(tf.constant([[1.,2.],[3.,4.]], dtype=tf.double)).graph
tf.io.write_graph(g, "./", "graph.pb", as_text=False)
```

Create a new go module with following `go.mod` file. As you can see
we have multiple `replace` clauses to pin to particular commit ID's.
```
module matrix-inverse

go 1.18

require github.com/tensorflow/tensorflow/tensorflow/go v0.0.0

require (
	github.com/kubetrail/tfutil v0.0.0-20220331172529-1c0569ffd50c // indirect
	google.golang.org/protobuf v1.28.0 // indirect
)

replace github.com/tensorflow/tensorflow/tensorflow/go => github.com/kubetrail-labs/tensorflow/tensorflow/go v0.0.0-20220330185145-9a3cb0962c98
```

Write `Go` code:
```go
package main

import (
	_ "embed"
	"fmt"
	"log"
	"math/rand"

	"github.com/kubetrail/tfutil"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

//go:embed models/graph.pb
var def []byte

func main() {
	// to invert a 5 x 5 matrix
	n := 5

	// generate random matrix input data
	input := make([]float64, n*n)
	for i := range input {
		input[i] = rand.Float64()
	}

	// create a tensor with shape n x n
	tensor, err := tfutil.NewTensor(input, n, n)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("input matrix:", tensor)

	// marshal tensor to upstream tf.Tensor object
	// which can be fed to tensorflow session
	x, err := tensor.Marshal()
	if err != nil {
		log.Fatal(err)
	}

	// import the graph
	g := tf.NewGraph()
	if err := g.Import(def, ""); err != nil {
		log.Fatal(err)
	}

	// prepare data feed specifying names of the operation.
	// The name x comes from python code that was used to
	// export the graph. See python code in models dir
	feeds := map[tf.Output]*tf.Tensor{
		g.Operation("input").Output(0): x,
	}

	// prepare data outputs to receive from tensorflow
	// session runs. The name Identity is the final
	// output of the graph
	fetches := []tf.Output{
		g.Operation("Identity").Output(0),
	}

	// start new session
	sess, err := tf.NewSession(
		g,
		&tf.SessionOptions{},
	)
	if err != nil {
		log.Fatal(err)
	}
	defer sess.Close()

	// run session
	out, err := sess.Run(feeds, fetches, nil)
	if err != nil {
		log.Fatal(err)
	}

	// since we expect just one output, ensure it is so
	if len(out) != 1 {
		log.Fatal("expected output to have 1 tensor, received", len(out))
	}

	// unmarshal output into matrix
	if err := tensor.Unmarshal(out[0]); err != nil {
		log.Fatal(err)
	}

	// print inverted matrix
	fmt.Println("inverted matrix:", tensor)
}
```

Running this code produces:
```bash
go run main.go 
input matrix: [[0.6046602879796196,0.9405090880450124,0.6645600532184904,0.4377141871869802,0.4246374970712657],[0.6868230728671094,0.06563701921747622,0.15651925473279124,0.09696951891448456,0.30091186058528707],[0.5152126285020654,0.8136399609900968,0.21426387258237492,0.380657189299686,0.31805817433032985],[0.4688898449024232,0.28303415118044517,0.29310185733681576,0.6790846759202163,0.21855305259276428],[0.20318687664732285,0.360871416856906,0.5706732760710226,0.8624914374478864,0.29311424455385804]]
inverted matrix: [[1.326302317049744,-0.13382937179671123,-1.5241040918148205,3.4709920537115804,-2.7182882092548732],[0.5647043565175299,-1.211411206833943,0.9216998312774841,0.4173078118858744,-0.885745783269239],[2.9600099759890797,-0.5993164120459926,-3.4009914688680962,1.1673572264105623,-0.8529307712748238],[-1.0256616804088645,-0.5547649814313621,0.57114234242179,1.3082838291342402,0.4601750301751597],[-4.359552916330332,4.383444275003038,4.8626513930315705,-9.042268391824125,6.692982813192693]]
```

Verify matrix inversion using `Julia`:
```
julia> input = [[0.6046602879796196,0.9405090880450124,0.6645600532184904,0.4377141871869802,0.4246374970712657],[0.6868230728671094,0.06563701921747622,0.15651925473279124,0.09696951891448456,0.30091186058528707],[0.5152126285020654,0.8136399609900968,0.21426387258237492,0.380657189299686,0.31805817433032985],[0.4688898449024232,0.28303415118044517,0.29310185733681576,0.6790846759202163,0.21855305259276428],[0.20318687664732285,0.360871416856906,0.5706732760710226,0.8624914374478864,0.29311424455385804]]
5-element Vector{Vector{Float64}}:
 [0.6046602879796196, 0.9405090880450124, 0.6645600532184904, 0.4377141871869802, 0.4246374970712657]
 [0.6868230728671094, 0.06563701921747622, 0.15651925473279124, 0.09696951891448456, 0.30091186058528707]
 [0.5152126285020654, 0.8136399609900968, 0.21426387258237492, 0.380657189299686, 0.31805817433032985]
 [0.4688898449024232, 0.28303415118044517, 0.29310185733681576, 0.6790846759202163, 0.21855305259276428]
 [0.20318687664732285, 0.360871416856906, 0.5706732760710226, 0.8624914374478864, 0.29311424455385804]

julia> x = reduce(vcat,transpose.(input))
5×5 Matrix{Float64}:
 0.60466   0.940509  0.66456   0.437714   0.424637
 0.686823  0.065637  0.156519  0.0969695  0.300912
 0.515213  0.81364   0.214264  0.380657   0.318058
 0.46889   0.283034  0.293102  0.679085   0.218553
 0.203187  0.360871  0.570673  0.862491   0.293114

julia> y = inv(x)
5×5 Matrix{Float64}:
  1.3263    -0.133829  -1.5241     3.47099   -2.71829
  0.564704  -1.21141    0.9217     0.417308  -0.885746
  2.96001   -0.599316  -3.40099    1.16736   -0.852931
 -1.02566   -0.554765   0.571142   1.30828    0.460175
 -4.35955    4.38344    4.86265   -9.04227    6.69298

julia> 
```

## shipping binary
Unlike pure Go binaries, the use of `libtensorflow.so` as a dynamic dependency
puts restrictions on where the compiled binary can run. The final deployment
package consists of `libtensorflow*.so` files, compiled Go binary and
environment variable `LD_LIBRARY_PATH`. Full description can be found
in the [Dockerfile](./examples/matrix-inverse/Dockerfile)

Build it as follows:
```bash
cd example/matrix-inverse
docker build -t tf-example ./
```

Run:
```bash
docker run --rm tf-example matrix-inverse
input matrix: [[0.6046602879796196,0.9405090880450124,0.6645600532184904,0.4377141871869802,0.4246374970712657],[0.6868230728671094,0.06563701921747622,0.15651925473279124,0.09696951891448456,0.30091186058528707],[0.5152126285020654,0.8136399609900968,0.21426387258237492,0.380657189299686,0.31805817433032985],[0.4688898449024232,0.28303415118044517,0.29310185733681576,0.6790846759202163,0.21855305259276428],[0.20318687664732285,0.360871416856906,0.5706732760710226,0.8624914374478864,0.29311424455385804]]
2022-03-31 20:41:00.923219: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
inverted matrix: [[1.326302317049744,-0.13382937179671123,-1.5241040918148205,3.4709920537115804,-2.7182882092548732],[0.5647043565175299,-1.211411206833943,0.9216998312774841,0.4173078118858744,-0.885745783269239],[2.9600099759890797,-0.5993164120459926,-3.4009914688680962,1.1673572264105623,-0.8529307712748238],[-1.0256616804088645,-0.5547649814313621,0.57114234242179,1.3082838291342402,0.4601750301751597],[-4.359552916330332,4.383444275003038,4.8626513930315705,-9.042268391824125,6.692982813192693]]
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
