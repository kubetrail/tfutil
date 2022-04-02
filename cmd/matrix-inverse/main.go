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
