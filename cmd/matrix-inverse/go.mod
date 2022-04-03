module matrix-inverse

go 1.18

require github.com/kubetrail/tfutil v0.0.0-20220403162045-9b280c99caa1

require (
	github.com/tensorflow/tensorflow/tensorflow/go v0.0.0 // indirect
	golang.org/x/exp v0.0.0-20220328175248-053ad81199eb // indirect
	google.golang.org/protobuf v1.28.0 // indirect
)

replace github.com/tensorflow/tensorflow/tensorflow/go => github.com/kubetrail-labs/tensorflow/tensorflow/go v0.0.0-20220330185145-9a3cb0962c98
