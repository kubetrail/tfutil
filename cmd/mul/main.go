package main

import (
	"fmt"
	"log"

	"github.com/kubetrail/tfutil/pkg/tfutil"
)

func main() {
	x, err := tfutil.NewTensor([]int32{1, 2, 3, 4}, 2, 2)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("matrix x:")
	fmt.Println(x)

	y, err := tfutil.Mul(x, x)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("matrix y = x*x:")
	fmt.Println(y)
}
