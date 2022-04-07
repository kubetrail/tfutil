module matrix-inverse

go 1.18

require github.com/kubetrail/tfutil v0.0.0

require (
	github.com/mattn/go-runewidth v0.0.9 // indirect
	github.com/olekukonko/tablewriter v0.0.5 // indirect
	github.com/tensorflow/tensorflow/tensorflow/go v0.0.0 // indirect
	golang.org/x/exp v0.0.0-20220328175248-053ad81199eb // indirect
	google.golang.org/protobuf v1.28.0 // indirect
)

replace (
	github.com/kubetrail/tfutil => ../../
	github.com/tensorflow/tensorflow/tensorflow/go => github.com/kubetrail-labs/tensorflow/tensorflow/go v0.0.0-20220330185145-9a3cb0962c98
)
