package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/kubetrail/tfutil/pkg/tfutil"
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

	fmt.Println("matrix:")
	fmt.Println(x)

	fmt.Println("matrix inverse:")
	fmt.Println(y)
}
