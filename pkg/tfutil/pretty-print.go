package tfutil

import (
	"bufio"
	"bytes"
	"fmt"

	"github.com/olekukonko/tablewriter"
)

// PrettyPrint prints matrices striding the tensor in chunks of
// first two dimensions along unit steps of remaining dimensions.
// For instance, a tensor with shape [p, q, r, s] will print r*s
// matrices, each with shape of [p, q]
func (g *Tensor[T]) PrettyPrint() ([]byte, error) {
	bb := &bytes.Buffer{}
	bw := bufio.NewWriter(bb)

	table := tablewriter.NewWriter(bw)
	table.SetBorder(false)
	table.SetColumnSeparator(" ")

	shape := g.shape
	switch len(shape) {
	case 0:
		return nil, nil
	case 1:
		if _, err := fmt.Fprintf(bw, "[ # vector shape: %v, dataType: %T\n", g.shape, *new(T)); err != nil {
			return nil, fmt.Errorf("failed to write to buffer: %w", err)
		}

		row := make([]string, len(g.value)+2)
		for i, v := range g.value {
			row[i+1] = fmt.Sprint(v)
		}
		table.Append(row)
		table.Render()

		if _, err := fmt.Fprintln(bw, "]"); err != nil {
			return nil, fmt.Errorf("failed to write to buffer: %w", err)
		}
	case 2:
		if _, err := fmt.Fprintf(bw, "[ # matrix shape: %v, dataType: %T\n", g.shape, *new(T)); err != nil {
			return nil, fmt.Errorf("failed to write to buffer: %w", err)
		}
		mdSlice, err := g.GetMultiDimSlice()
		if err != nil {
			return nil, fmt.Errorf("failed to get multi dim slice: %w", err)
		}
		mat, ok := mdSlice.([][]T)
		if !ok {
			return nil, fmt.Errorf("failed to type assert on multi dim slice")
		}

		for _, row := range mat {
			r := make([]string, len(row)+2)
			for q, v := range row {
				r[q+1] = fmt.Sprintf(" %v ", v)
			}
			table.Append(r)
		}

		table.Render()

		if _, err := fmt.Fprintln(bw, "]"); err != nil {
			return nil, fmt.Errorf("failed to write to buffer: %w", err)
		}
	default:
		start := make([]int, len(g.shape))
		end := clone(g.shape)
		for i := 0; i < g.shape[len(g.shape)-1]; i++ {
			start[len(start)-1] = i
			end[len(end)-1] = i + 1

			if _, err := fmt.Fprintf(bw,
				"[ # sub tensor start: %v, end: %v, reshpaed: %v\n",
				start, end, g.shape[:len(g.shape)-1],
			); err != nil {
				return nil, err
			}

			sub, err := g.Sub(start, end, nil)
			if err != nil {
				return nil, fmt.Errorf("failed to get sub tensor: %w", err)
			}

			if err := sub.Reshape(g.shape[:len(g.shape)-1]...); err != nil {
				return nil, fmt.Errorf("failed to reshape: %w", err)
			}

			b, err := sub.PrettyPrint()
			if err != nil {
				return nil, err
			}

			if _, err := bw.Write(b); err != nil {
				return nil, err
			}

			if _, err := fmt.Fprintln(bw, "]"); err != nil {
				return nil, err
			}
		}
	}

	if err := bw.Flush(); err != nil {
		return nil, fmt.Errorf("failed to flush writer: %w", err)
	}

	return bb.Bytes(), nil
}
