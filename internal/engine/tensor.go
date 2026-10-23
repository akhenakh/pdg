package engine

import (
	"fmt"
	"math"
)

// Kind is the storage type of a tensor. The model uses float32, int64
// (booleans are 0/1 integer tensors) and float64 for its phase/time grid.
type Kind uint8

const (
	F32 Kind = iota
	I64
	F64
)

func (k Kind) String() string {
	switch k {
	case F32:
		return "f32"
	case F64:
		return "f64"
	default:
		return "i64"
	}
}

// Tensor is a dense, row-major tensor.
type Tensor struct {
	Kind  Kind
	Shape []int
	F     []float32
	F64   []float64
	I     []int64
}

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func newTensor(kind Kind, shape []int) *Tensor {
	n := numel(shape)
	s := append([]int(nil), shape...)
	t := &Tensor{Kind: kind, Shape: s}
	switch kind {
	case F32:
		t.F = make([]float32, n)
	case F64:
		t.F64 = make([]float64, n)
	default:
		t.I = make([]int64, n)
	}
	return t
}

func newF32(shape ...int) *Tensor { return newTensor(F32, shape) }
func newF64(shape ...int) *Tensor { return newTensor(F64, shape) }
func newI64(shape ...int) *Tensor { return newTensor(I64, shape) }

func (t *Tensor) Numel() int { return numel(t.Shape) }
func (t *Tensor) Rank() int  { return len(t.Shape) }

// at reads element i as float64 (integers are converted).
func (t *Tensor) at(i int) float64 {
	switch t.Kind {
	case F32:
		return float64(t.F[i])
	case F64:
		return t.F64[i]
	default:
		return float64(t.I[i])
	}
}

// set writes element i, converting as needed.
func (t *Tensor) set(i int, v float64) {
	switch t.Kind {
	case F32:
		t.F[i] = float32(v)
	case F64:
		t.F64[i] = v
	default:
		t.I[i] = int64(v)
	}
}

// copyElems copies n elements from src[si:] into dst[di:].
func copyElems(dst *Tensor, di int, src *Tensor, si, n int) {
	if dst.Kind == src.Kind {
		switch dst.Kind {
		case F32:
			copy(dst.F[di:di+n], src.F[si:si+n])
			return
		case F64:
			copy(dst.F64[di:di+n], src.F64[si:si+n])
			return
		default:
			copy(dst.I[di:di+n], src.I[si:si+n])
			return
		}
	}
	for k := 0; k < n; k++ {
		dst.set(di+k, src.at(si+k))
	}
}

func strides(shape []int) []int {
	s := make([]int, len(shape))
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = acc
		acc *= shape[i]
	}
	return s
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// batchStrides returns element strides for the leading batch dimensions of
// a tensor with the given full shape, projected/broadcast onto batchShape.
func batchStrides(fullShape, batchShape []int) []int {
	st := strides(fullShape)
	bdims := len(fullShape) - 2
	res := make([]int, len(batchShape))
	off := len(batchShape) - bdims
	for i := range batchShape {
		if i < off {
			res[i] = 0
			continue
		}
		d := fullShape[i-off]
		if d == 1 && batchShape[i] != 1 {
			res[i] = 0
		} else {
			res[i] = st[i-off]
		}
	}
	return res
}

// broadcastShape returns the numpy-style broadcast of two shapes.
func broadcastShape(a, b []int) ([]int, error) {
	r := len(a)
	if len(b) > r {
		r = len(b)
	}
	out := make([]int, r)
	for i := 0; i < r; i++ {
		da, db := 1, 1
		if j := i - (r - len(a)); j >= 0 {
			da = a[j]
		}
		if j := i - (r - len(b)); j >= 0 {
			db = b[j]
		}
		switch {
		case da == db || db == 1:
			out[i] = da
		case da == 1:
			out[i] = db
		default:
			return nil, fmt.Errorf("shapes %v and %v are not broadcastable", a, b)
		}
	}
	return out, nil
}

// broadcastStrides projects shape onto out, using stride 0 for broadcast axes.
func broadcastStrides(shape, out []int) []int {
	st := strides(shape)
	res := make([]int, len(out))
	off := len(out) - len(shape)
	for i := range out {
		if i < off {
			res[i] = 0
		} else {
			d := shape[i-off]
			if d == 1 && out[i] != 1 {
				res[i] = 0
			} else {
				res[i] = st[i-off]
			}
		}
	}
	return res
}

// binOp applies fn elementwise with broadcasting.
func binOp(a, b *Tensor, fn func(float64, float64) float64) (*Tensor, error) {
	shape, err := broadcastShape(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	if a.Kind != b.Kind {
		return nil, fmt.Errorf("mixed kinds %s and %s", a.Kind, b.Kind)
	}
	out := newTensor(a.Kind, shape)
	if sameShape(a.Shape, shape) && sameShape(b.Shape, shape) {
		switch a.Kind {
		case F32:
			for i := range out.F {
				out.F[i] = float32(fn(float64(a.F[i]), float64(b.F[i])))
			}
		case F64:
			for i := range out.F64 {
				out.F64[i] = fn(a.F64[i], b.F64[i])
			}
		default:
			for i := range out.I {
				out.I[i] = int64(fn(float64(a.I[i]), float64(b.I[i])))
			}
		}
		return out, nil
	}
	sa := broadcastStrides(a.Shape, shape)
	sb := broadcastStrides(b.Shape, shape)
	idx := make([]int, len(shape))
	for i := 0; i < out.Numel(); i++ {
		oa, ob := 0, 0
		for d := range shape {
			oa += idx[d] * sa[d]
			ob += idx[d] * sb[d]
		}
		out.set(i, fn(a.at(oa), b.at(ob)))
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out, nil
}

func compareTensors(a, b *Tensor, fn func(float64, float64) bool) (*Tensor, error) {
	shape, err := broadcastShape(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	out := newTensor(I64, shape)
	sa := broadcastStrides(a.Shape, shape)
	sb := broadcastStrides(b.Shape, shape)
	idx := make([]int, len(shape))
	for i := 0; i < out.Numel(); i++ {
		oa, ob := 0, 0
		for d := range shape {
			oa += idx[d] * sa[d]
			ob += idx[d] * sb[d]
		}
		if fn(a.at(oa), b.at(ob)) {
			out.I[i] = 1
		}
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out, nil
}

func scalarF32(t *Tensor) (float32, error) {
	if t.Numel() != 1 {
		return 0, fmt.Errorf("expected scalar, got shape %v", t.Shape)
	}
	return float32(t.at(0)), nil
}

func scalarI64(t *Tensor) (int64, error) {
	if t.Numel() != 1 {
		return 0, fmt.Errorf("expected scalar, got shape %v", t.Shape)
	}
	return int64(t.at(0)), nil
}

var _ = math.Abs
