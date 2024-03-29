package tfutil

import (
	"bufio"
	"bytes"
	"fmt"

	"github.com/olekukonko/tablewriter"
)

// String prints matrices in human-readable format
func (tensor *Tensor[T]) String() string {
	bb := &bytes.Buffer{}
	bw := bufio.NewWriter(bb)

	table := tablewriter.NewWriter(bw)
	table.SetBorder(false)
	table.SetColumnSeparator(" ")

	shape := tensor.shape
	switch len(shape) {
	case 0:
		return ""
	case 1:
		if _, err := fmt.Fprintf(bw, "[ # vector shape: %v, dataType: %T\n", tensor.shape, *new(T)); err != nil {
			return fmt.Errorf("failed to write to buffer: %w", err).Error()
		}

		row := make([]string, len(tensor.value)+2)
		for i, v := range tensor.value {
			row[i+1] = fmt.Sprint(v)
		}
		table.Append(row)
		table.Render()

		if _, err := fmt.Fprintln(bw, "]"); err != nil {
			return fmt.Errorf("failed to write to buffer: %w", err).Error()
		}
	case 2:
		if _, err := fmt.Fprintf(bw, "[ # matrix shape: %v, dataType: %T\n", tensor.shape, *new(T)); err != nil {
			return fmt.Errorf("failed to write to buffer: %w", err).Error()
		}
		mdSlice, err := tensor.GetMultiDimSlice()
		if err != nil {
			return fmt.Errorf("failed to get multi dim slice: %w", err).Error()
		}
		mat, ok := mdSlice.([][]T)
		if !ok {
			return fmt.Errorf("failed to type assert on multi dim slice").Error()
		}

		for _, row := range mat {
			r := make([]string, len(row)+2)
			for q, v := range row {
				r[q+1] = fmt.Sprint(v)
			}
			table.Append(r)
		}

		table.Render()

		if _, err := fmt.Fprintln(bw, "]"); err != nil {
			return fmt.Errorf("failed to write to buffer: %w", err).Error()
		}
	default:
		start := make([]int, len(tensor.shape))
		end := clone(tensor.shape)
		for i := 0; i < tensor.shape[len(tensor.shape)-1]; i++ {
			start[len(start)-1] = i
			end[len(end)-1] = i + 1

			if _, err := fmt.Fprintf(bw,
				"[ # sub tensor start: %v, end: %v, reshpaed: %v\n",
				start, end, tensor.shape[:len(tensor.shape)-1],
			); err != nil {
				return err.Error()
			}

			sub, err := tensor.Sub(start, end, nil)
			if err != nil {
				return fmt.Errorf("failed to get sub tensor: %w", err).Error()
			}

			if err := sub.Reshape(tensor.shape[:len(tensor.shape)-1]...); err != nil {
				return fmt.Errorf("failed to reshape: %w", err).Error()
			}

			b := sub.String()

			if _, err := bw.Write([]byte(b)); err != nil {
				return fmt.Errorf("failed to write to output buffer: %w", err).Error()
			}

			if _, err := fmt.Fprintln(bw, "]"); err != nil {
				return fmt.Errorf("failed to write to output buffer: %w", err).Error()
			}
		}
	}

	if err := bw.Flush(); err != nil {
		return fmt.Errorf("failed to flush writer: %w", err).Error()
	}

	return bb.String()
}
