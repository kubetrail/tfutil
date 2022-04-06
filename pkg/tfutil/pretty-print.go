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

	if _, err := fmt.Fprintf(bw, "shape: %v, dataType: %T\n", g.shape, *new(T)); err != nil {
		return nil, fmt.Errorf("failed to write to buffer: %w", err)
	}

	table := tablewriter.NewWriter(bw)
	table.SetBorder(false)
	table.SetColumnSeparator(" ")

	shape := g.shape
	switch len(shape) {
	case 0:
		return nil, nil
	case 1:
		table.SetHeader([]string{fmt.Sprintf("shape:%v, dataType:%T", g.shape, *new(T))})
		row := make([]string, len(g.value))
		for i, v := range g.value {
			row[i] = fmt.Sprint(v)
		}
		table.Append(row)
	case 2:
		table.SetHeader([]string{fmt.Sprintf("shape:%v, dataType:%T", g.shape, *new(T))})
		mdSlice, err := g.GetMultiDimSlice()
		if err != nil {
			return nil, fmt.Errorf("failed to get multi dim slice: %w", err)
		}
		mat, ok := mdSlice.([][]T)
		if !ok {
			return nil, fmt.Errorf("failed to type assert on multi dim slice")
		}

		r := make([]string, g.shape[1]+2)
		r[0] = "["
		table.Append(r)

		for _, row := range mat {
			r := make([]string, len(row)+2)
			for q, v := range row {
				r[q+1] = fmt.Sprintf(" %v ", v)
			}
			table.Append(r)
		}

		r = make([]string, g.shape[1]+2)
		r[len(r)-1] = "]"
		table.Append(r)
	default:
		n, err := numElements(g.shape[2:])
		if err != nil {
			return nil, fmt.Errorf("failed to collapse dimensions: %w", err)
		}

		if _, err := fmt.Fprintf(bw, "printing %d matrices sequentially, each with shape: %v\n",
			n, []int{g.shape[0], g.shape[1]}); err != nil {
			return nil, fmt.Errorf("failed to write to buffer: %w", err)
		}

		x, err := g.Clone()
		if err != nil {
			return nil, fmt.Errorf("failed to clone tensor: %w", err)
		}

		if err := x.Reshape([]int{n, g.shape[0], g.shape[1]}...); err != nil {
			return nil, fmt.Errorf("failed to reshape tensor: %w", err)
		}

		r := make([]string, g.shape[1]+2)
		r[0] = "["
		table.Append(r)
		for i := 0; i < n; i++ {
			sub, err := x.Sub([]int{i, 0, 0}, []int{i + 1, g.shape[0], g.shape[1]}, nil)
			if err != nil {
				return nil, fmt.Errorf("failed to get sub tensor: %w", err)
			}

			if err := sub.Reshape(g.shape[0], g.shape[1]); err != nil {
				return nil, fmt.Errorf("failed to reshape: %w", err)
			}

			mdSlice, err := sub.GetMultiDimSlice()
			if err != nil {
				return nil, fmt.Errorf("failed to get multi dim slice: %w", err)
			}

			mat, ok := mdSlice.([][]T)
			if !ok {
				return nil, fmt.Errorf("failed to type assert on multi dim slice")
			}

			r := make([]string, g.shape[1]+2)
			r[0] = "["
			table.Append(r)

			for _, row := range mat {
				r := make([]string, len(row)+2)
				for q, v := range row {
					r[q+1] = fmt.Sprintf(" %v ", v)
				}
				table.Append(r)
			}

			r = make([]string, g.shape[1]+2)
			r[len(r)-1] = "]"
			table.Append(r)
		}
		r = make([]string, g.shape[1]+2)
		r[len(r)-1] = "]"
		table.Append(r)
	}

	table.Render()

	if err := bw.Flush(); err != nil {
		return nil, fmt.Errorf("failed to flush writer: %w", err)
	}

	return bb.Bytes(), nil
}
