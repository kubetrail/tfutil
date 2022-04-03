package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/kubetrail/tfutil/pkg/tfutil"
)

func main() {
	// to invert a 5 x 5 matrix
	n := 5

	// generate random matrix input data
	input := make([]float64, n*n)
	for i := range input {
		input[i] = rand.Float64()
	}

	// create a matrix with shape n x n
	x, err := tfutil.NewTensor(input, n, n)
	if err != nil {
		log.Fatal(err)
	}

	y, err := tfutil.MatrixInverse(x)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("input matrix:", x)
	fmt.Println("inverted matrix:", y)
}
