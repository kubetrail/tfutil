module matrix-inverse

go 1.18

require github.com/kubetrail/tfutil v0.0.0

require (
	github.com/mattn/go-runewidth v0.0.9 // indirect
	github.com/olekukonko/tablewriter v0.0.5 // indirect
	golang.org/x/exp v0.0.0-20220328175248-053ad81199eb // indirect
	google.golang.org/protobuf v1.28.0 // indirect
	github.com/wamuir/graft v0.4.0
)

replace (
	github.com/kubetrail/tfutil => ../../
)
