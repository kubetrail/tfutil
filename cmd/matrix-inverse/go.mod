module matrix-inverse

go 1.18

require github.com/kubetrail/tfutil v0.0.0-20220402233800-453eae5c8d53

require (
	github.com/tensorflow/tensorflow/tensorflow/go v0.0.0 // indirect
	google.golang.org/protobuf v1.28.0 // indirect
)

replace github.com/tensorflow/tensorflow/tensorflow/go => github.com/kubetrail-labs/tensorflow/tensorflow/go v0.0.0-20220330185145-9a3cb0962c98
