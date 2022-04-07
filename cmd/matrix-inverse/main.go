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
