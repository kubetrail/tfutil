package main

import (
	"archive/tar"
	"compress/gzip"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/kubetrail/tfutil/pkg/proto/graph"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func getLabels() ([]string, error) {
	labelsFile := "imagenet_slim_labels.txt"
	f, err := os.Stat(labelsFile)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			resp, err := http.Get("https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz")
			if err != nil {
				return nil, fmt.Errorf("failed to download graph: %w", err)
			}

			gr, err := gzip.NewReader(resp.Body)
			if err != nil {
				return nil, fmt.Errorf("failed to create gzip reader: %w", err)
			}
			defer resp.Body.Close()

			tr := tar.NewReader(gr)
			for {
				header, err := tr.Next()
				if err == io.EOF {
					break
				}

				if err != nil {
					return nil, fmt.Errorf("failed to extract tar bundle: %w", err)
				}

				switch header.Typeflag {
				case tar.TypeDir:
					if err := os.Mkdir(header.Name, 0755); err != nil {
						return nil, fmt.Errorf("failed to extract tar bundle creating dir: %w", err)
					}
				case tar.TypeReg:
					outFile, err := os.Create(header.Name)
					if err != nil {
						return nil, fmt.Errorf("failed to extract tar bundle creating file: %w", err)
					}
					if _, err := io.Copy(outFile, tr); err != nil {
						return nil, fmt.Errorf("failed to extract tar bundle writing file: %w", err)
					}
					if err := outFile.Close(); err != nil {
						return nil, fmt.Errorf("failed to extract tar bundle closing file: %w", err)
					}
				default:
					return nil, fmt.Errorf("failed to extract tar bundle, unknown header")
				}
			}
		} else {
			return nil, fmt.Errorf("failed to stat file: %w", err)
		}
	}

	if f.IsDir() {
		return nil, fmt.Errorf("a folder by the name %s exists", labelsFile)
	}

	b, err := os.ReadFile(labelsFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read labels file: %w", err)
	}

	labels := strings.Split(string(b), "\n")
	return labels, nil
}

func loadImage(fileName string, t *testing.T) {
	if _, err := os.Stat(fileName); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			resp, err := http.Get("https://raw.githubusercontent.com/tensorflow/tensorflow/v2.8.0/tensorflow/examples/label_image/data/grace_hopper.jpg")
			if err != nil {
				t.Fatal(err)
			}

			b, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}

			if err := resp.Body.Close(); err != nil {
				t.Fatal(err)
			}

			if err := os.WriteFile(fileName, b, 0644); err != nil {
				t.Fatal(err)
			}

		} else {
			t.Fatalf("failed to stat image file: %s", err)
		}
	}
}

// TestImageClassificationTwoSteps tests image classification in two
// steps. In first step, a Go API based tensorflow graph is constructed
// that outputs a resized and scaled image tensor of the shape
// [1 299 299 3]. Such tensor is then fed in the second step
// to upstream frozen pretrained graph.
func TestImageClassificationTwoSteps(t *testing.T) {
	labels, err := getLabels()
	if err != nil {
		t.Fatal(err)
	}

	fileName := "grace_hopper.jpg"
	loadImage(fileName, t)

	imageName, err := tf.NewTensor(fileName)
	if err != nil {
		t.Fatal(err)
	}

	root := op.NewScope().SubScope("image")
	ns := func(namespace string) *op.Scope {
		return root.SubScope(namespace)
	}

	ImageName := op.Placeholder(ns("imageName"), tf.String)
	ImageRaw := op.ReadFile(ns("imageRaw"), ImageName)
	ImageDecoded := op.DecodePng(ns("imageDecoded"), ImageRaw, op.DecodePngChannels(3))
	ImageCast := op.Cast(ns("imageCast"), ImageDecoded, tf.Float)
	ImageOffset := op.Sub(ns("imageOffset"), ImageCast, op.Const(ns("imageOffset"), float32(128)))
	ImageScaled := op.Div(ns("imageScaled"), ImageOffset, op.Const(ns("imageScaled"), float32(255)))
	ImagesBatched := op.ExpandDims(ns("imagesBatched"), ImageScaled, op.Const(ns("expandDimsDim"), int64(0)))
	ImagesResized := op.ResizeBilinear(ns("resizedImages"), ImagesBatched, op.Const(ns("resizedImageShape"), []int32{299, 299}))

	tfGraph, err := root.Finalize()
	if err != nil {
		t.Fatal(err)
	}

	session, err := tf.NewSession(tfGraph, nil)
	if err != nil {
		t.Fatal(err)
	}

	feeds := map[tf.Output]*tf.Tensor{
		ImageName: imageName,
	}

	outTensors, err := session.Run(feeds, []tf.Output{ImagesResized}, nil)
	if err != nil {
		t.Fatal(err)
	}
	session.Close()

	graphDef, err := graph.LoadFile("inception_v3_2016_08_28_frozen.pb")
	if err != nil {
		t.Fatal(err)
	}

	tfGraph, err = graphDef.Export("")
	if err != nil {
		t.Fatal(err)
	}

	feeds = map[tf.Output]*tf.Tensor{
		tfGraph.Operation("input").Output(0): outTensors[0],
	}

	fetches := []tf.Output{
		tfGraph.Operation("InceptionV3/Predictions/Reshape_1").Output(0),
	}

	session, err = tf.NewSession(tfGraph, nil)
	if err != nil {
		t.Fatal(err)
	}

	outTensors, err = session.Run(feeds, fetches, nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	labelValues, ok := (outTensors[0].Value()).([][]float32)
	if !ok {
		t.Fatalf("expected out tensors to be type assertable to [][]float32, got %T", outTensors[0].Value())
	}

	maxIndex := -1
	maxValue := float32(-1)
	for i, v := range labelValues[0] {
		if v > maxValue {
			maxValue = v
			maxIndex = i
		}
	}

	if maxIndex >= 0 {
		fmt.Println(labels[maxIndex])
	}
}

// TestIntegrateGraphs adds a few extra nodes to the upstream
// pretrained frozen graph in order for it to work directly
// using a filename as an input. The integrated graph is written
// to the same folder in three separate formats although
// only binary formatted graph will be required in subsequent tests
func TestIntegrateGraphs(t *testing.T) {
	labels, err := getLabels()
	if err != nil {
		t.Fatal(err)
	}

	fileName := "grace_hopper.jpg"
	loadImage(fileName, t)

	imageName, err := tf.NewTensor(fileName)
	if err != nil {
		t.Fatal(err)
	}

	root := op.NewScope().SubScope("image")
	ns := func(namespace string) *op.Scope {
		return root.SubScope(namespace)
	}

	ImageName := op.Placeholder(ns("imageName"), tf.String)
	ImageRaw := op.ReadFile(ns("imageRaw"), ImageName)
	ImageDecoded := op.DecodeImage(ns("imageDecoded"), ImageRaw)
	//ImageDecoded := op.DecodePng(ns("imageDecoded"), ImageRaw, op.DecodePngChannels(3))
	ImageCast := op.Cast(ns("imageCast"), ImageDecoded, tf.Float)
	ImageOffset := op.Sub(ns("imageOffset"), ImageCast, op.Const(ns("imageOffset"), float32(128)))
	ImageScaled := op.Div(ns("imageScaled"), ImageOffset, op.Const(ns("imageScaled"), float32(255)))
	ImagesBatched := op.ExpandDims(ns("imagesBatched"), ImageScaled, op.Const(ns("expandDimsDim"), int64(0)))
	_ = op.ResizeBilinear(ns("resizedImages"), ImagesBatched, op.Const(ns("resizedImageShape"), []int32{299, 299}))

	tfGraph, err := root.Finalize()
	if err != nil {
		t.Fatal(err)
	}

	graphDefImageLoad, err := graph.NewGraphDef()
	if err != nil {
		t.Fatal(err)
	}

	if err := graphDefImageLoad.Import(tfGraph); err != nil {
		t.Fatal(err)
	}

	if err := graphDefImageLoad.ApplyPrefix("integrated"); err != nil {
		t.Fatal(err)
	}

	graphDef, err := graph.LoadFile("inception_v3_2016_08_28_frozen.pb")
	if err != nil {
		t.Fatal(err)
	}

	if err := graphDef.Append(graphDefImageLoad); err != nil {
		t.Fatal(err)
	}

	if err := graphDef.RenameNode("input", "oldInput"); err != nil {
		t.Fatal(err)
	}

	if err := graphDef.RenameNode("integrated/image/resizedImages/ResizeBilinear", "input"); err != nil {
		t.Fatal(err)
	}

	if err := graph.SaveFile(graphDef, "integrated-graph.json"); err != nil {
		t.Fatal(err)
	}

	if err := graph.SaveFile(graphDef, "integrated-graph.pbtxt"); err != nil {
		t.Fatal(err)
	}

	if err := graph.SaveFile(graphDef, "integrated-graph.pb"); err != nil {
		t.Fatal(err)
	}

	tfGraph, err = graphDef.Export("")
	if err != nil {
		t.Fatal(err)
	}

	feeds := map[tf.Output]*tf.Tensor{
		tfGraph.Operation("integrated/image/imageName/Placeholder").Output(0): imageName,
	}

	fetches := []tf.Output{
		tfGraph.Operation("InceptionV3/Predictions/Reshape_1").Output(0),
	}

	session, err := tf.NewSession(tfGraph, nil)
	if err != nil {
		t.Fatal(err)
	}

	outTensors, err := session.Run(feeds, fetches, nil)
	if err != nil {
		t.Fatal(err)
	}
	session.Close()

	labelValues, ok := (outTensors[0].Value()).([][]float32)
	if !ok {
		t.Fatalf("expected out tensors to be type assertable to [][]float32, got %T", outTensors[0].Value())
	}

	maxIndex := -1
	maxValue := float32(-1)
	for i, v := range labelValues[0] {
		if v > maxValue {
			maxValue = v
			maxIndex = i
		}
	}

	if maxIndex >= 0 {
		fmt.Println(labels[maxIndex])
	}
}

// TestClassifyImageUsingIntegratedGraph tests image classification
// using integrated graph such that the only input to it is the
// filename.
func TestClassifyImageUsingIntegratedGraph(t *testing.T) {
	labels, err := getLabels()
	if err != nil {
		t.Fatal(err)
	}

	graphDef, err := graph.LoadFile("integrated-graph.pb")
	if err != nil {
		t.Fatal(err)
	}

	tfGraph, err := graphDef.Export("")
	if err != nil {
		t.Fatal(err)
	}

	fileName := "grace_hopper.jpg"
	loadImage(fileName, t)

	imageName, err := tf.NewTensor(fileName)
	if err != nil {
		t.Fatal(err)
	}

	feeds := map[tf.Output]*tf.Tensor{
		tfGraph.Operation("integrated/image/imageName/Placeholder").Output(0): imageName,
	}

	fetches := []tf.Output{
		tfGraph.Operation("InceptionV3/Predictions/Reshape_1").Output(0),
	}

	session, err := tf.NewSession(tfGraph, nil)
	if err != nil {
		t.Fatal(err)
	}

	outTensors, err := session.Run(feeds, fetches, nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	labelValues, ok := (outTensors[0].Value()).([][]float32)
	if !ok {
		t.Fatalf("expected out tensors to be type assertable to [][]float32, got %T", outTensors[0].Value())
	}

	maxIndex := -1
	maxValue := float32(-1)
	for i, v := range labelValues[0] {
		if v > maxValue {
			maxValue = v
			maxIndex = i
		}
	}

	if maxIndex >= 0 {
		fmt.Println(fileName, ":", labels[maxIndex])
	}
}
