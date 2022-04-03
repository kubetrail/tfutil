module github.com/kubetrail/tfutil

go 1.18

require (
	github.com/tensorflow/tensorflow/tensorflow/go v0.0.0
	golang.org/x/exp v0.0.0-20220328175248-053ad81199eb
)

require google.golang.org/protobuf v1.28.0 // indirect

// find out latest commit for branch v2.8.0-proto
// somewhere in another new clean module do:
// go get github.com/kubetrail-labs/tensorflow/tensorflow/go@the-commit-id
// for instance: go get github.com/kubetrail-labs/tensorflow/tensorflow/go@9a3cb0962c9
// then from the output (shown below in next line) copy the date timestamp and git commit id in replace clause
// go: downloading github.com/kubetrail-labs/tensorflow/tensorflow/go v0.0.0-20220330185145-9a3cb0962c98
replace github.com/tensorflow/tensorflow/tensorflow/go => github.com/kubetrail-labs/tensorflow/tensorflow/go v0.0.0-20220330185145-9a3cb0962c98
