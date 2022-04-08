package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/kubetrail/tfutil/pkg/tfutil"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	// create a random square matrix using a generator func
	x, err := tfutil.NewTensorFromFunc(
		func(int) float64 { return rand.Float64() }, 5, 5)
	if err != nil {
		log.Fatal(err)
	}

	// wrap op.MatrixInverse operator in a func literal
	// and apply on input x
	y, err := tfutil.ApplyOperators(
		x,
		func(scope *op.Scope, x tf.Output) tf.Output {
			return op.MatrixInverse(scope, x)
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	// wrap op.MatMul operator in a func literal and
	// apply on inputs x and y
	z, err := tfutil.ApplyOperatorXY(
		x, y,
		func(scope *op.Scope, x, y tf.Output) tf.Output {
			return op.MatMul(scope, x, y)
		},
	)
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
