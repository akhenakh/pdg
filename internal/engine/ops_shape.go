package engine

import (
	"fmt"
	"math"
)

func shapeNode(g *Graph, n *Node, env map[string]*Tensor) error {
	t := env[n.In[0]]
	rank := len(t.Shape)
	start := n.intAttr("start", 0)
	end := n.intAttr("end", rank)
	if start < 0 {
		start += rank
	}
	if end < 0 {
		end += rank
	}
	if start < 0 {
		start = 0
	}
	if end > rank {
		end = rank
	}
	if end < start {
		end = start
	}
	out := newI64(end - start)
	for i := start; i < end; i++ {
		out.I[i-start] = int64(t.Shape[i])
	}
	env[n.Out[0]] = out
	return nil
}

func reshapeNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	st := env[n.In[1]]
	if st.Kind != I64 {
		return fmt.Errorf("shape input must be int64")
	}
	allowzero := n.intAttr("allowzero", 0) != 0
	target := make([]int, len(st.I))
	for i, v := range st.I {
		target[i] = int(v)
	}
	neg := -1
	known := 1
	for i, d := range target {
		switch {
		case d == -1:
			if neg >= 0 {
				return fmt.Errorf("more than one -1 in shape %v", target)
			}
			neg = i
		case d == 0 && !allowzero:
			if i >= len(x.Shape) {
				return fmt.Errorf("0 at axis %d has no matching input dimension", i)
			}
			target[i] = x.Shape[i]
			known *= target[i]
		case d < 0:
			return fmt.Errorf("invalid dimension %d", d)
		default:
			known *= d
		}
	}
	if neg >= 0 {
		if known == 0 || x.Numel()%known != 0 {
			return fmt.Errorf("cannot infer dimension: %v has %d elements", target, x.Numel())
		}
		target[neg] = x.Numel() / known
	}
	if numel(target) != x.Numel() {
		return fmt.Errorf("reshape %v to %v changes element count %d -> %d", x.Shape, target, x.Numel(), numel(target))
	}
	out := &Tensor{Kind: x.Kind, Shape: target, F: x.F, F64: x.F64, I: x.I}
	env[n.Out[0]] = out
	return nil
}

func transposeNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	rank := len(x.Shape)
	perm := n.intsAttr("perm")
	if perm == nil {
		perm = make([]int, rank)
		for i := range perm {
			perm[i] = rank - 1 - i
		}
	}
	if len(perm) != rank {
		return fmt.Errorf("perm %v has %d axes, input has %d", perm, len(perm), rank)
	}
	outShape := make([]int, rank)
	for i, p := range perm {
		if p < 0 || p >= rank {
			return fmt.Errorf("bad perm %v", perm)
		}
		outShape[i] = x.Shape[p]
	}
	out := newTensor(x.Kind, outShape)
	src := strides(x.Shape)
	idx := make([]int, rank)
	for i := 0; i < out.Numel(); i++ {
		off := 0
		for d := 0; d < rank; d++ {
			off += idx[d] * src[perm[d]]
		}
		out.set(i, x.at(off))
		for d := rank - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < outShape[d] {
				break
			}
			idx[d] = 0
		}
	}
	env[n.Out[0]] = out
	return nil
}

func squeezeNode(n *Node, env map[string]*Tensor, squeeze bool) error {
	x := env[n.In[0]]
	var axes []int
	if len(n.In) > 1 && n.In[1] != "" {
		at := env[n.In[1]]
		if at.Kind != I64 {
			return fmt.Errorf("axes must be int64")
		}
		axes = make([]int, len(at.I))
		for i, v := range at.I {
			axes[i] = int(v)
		}
	} else {
		axes = n.intsAttr("axes")
	}
	if squeeze {
		rank := len(x.Shape)
		drop := make(map[int]bool)
		if axes == nil {
			for i, d := range x.Shape {
				if d == 1 {
					drop[i] = true
				}
			}
		} else {
			for _, a := range axes {
				if a < 0 {
					a += rank
				}
				if a < 0 || a >= rank {
					return fmt.Errorf("axis %d out of range", a)
				}
				if x.Shape[a] != 1 {
					return fmt.Errorf("cannot squeeze axis %d of size %d", a, x.Shape[a])
				}
				drop[a] = true
			}
		}
		outShape := make([]int, 0, rank)
		for i, d := range x.Shape {
			if !drop[i] {
				outShape = append(outShape, d)
			}
		}
		env[n.Out[0]] = &Tensor{Kind: x.Kind, Shape: outShape, F: x.F, F64: x.F64, I: x.I}
		return nil
	}
	if len(axes) == 0 {
		return fmt.Errorf("Unsqueeze without axes")
	}
	outRank := len(x.Shape) + len(axes)
	ins := make([]int, len(axes))
	for i, a := range axes {
		if a < 0 {
			a += outRank
		}
		if a < 0 || a >= outRank {
			return fmt.Errorf("axis %d out of range for rank %d", a, outRank)
		}
		ins[i] = a
	}
	for i := 1; i < len(ins); i++ {
		for j := i; j > 0 && ins[j] < ins[j-1]; j-- {
			ins[j], ins[j-1] = ins[j-1], ins[j]
		}
	}
	outShape := make([]int, 0, outRank)
	pos := 0
	for i := 0; i < outRank; i++ {
		if pos < len(ins) && ins[pos] == i {
			outShape = append(outShape, 1)
			pos++
		} else {
			k := len(outShape) - pos
			if k >= len(x.Shape) {
				return fmt.Errorf("bad axes %v", axes)
			}
			outShape = append(outShape, x.Shape[k])
		}
	}
	env[n.Out[0]] = &Tensor{Kind: x.Kind, Shape: outShape, F: x.F, F64: x.F64, I: x.I}
	return nil
}

func concatNode(n *Node, env map[string]*Tensor) error {
	axis := n.intAttr("axis", 0)
	var kind Kind
	var shape []int
	rank := -1
	for i, name := range n.In {
		if name == "" {
			continue
		}
		t := env[name]
		if i == 0 {
			kind = t.Kind
			rank = len(t.Shape)
			shape = append([]int(nil), t.Shape...)
			if axis < 0 {
				axis += rank
			}
			if axis < 0 || axis >= rank {
				return fmt.Errorf("bad concat axis %d", axis)
			}
			shape[axis] = 0
		}
		if t.Kind != kind || len(t.Shape) != rank {
			return fmt.Errorf("concat inputs have inconsistent shapes")
		}
		shape[axis] += t.Shape[axis]
	}
	out := newTensor(kind, shape)
	inner := numel(shape[axis+1:])
	outer := numel(shape[:axis])
	off := 0
	for _, name := range n.In {
		if name == "" {
			continue
		}
		t := env[name]
		chunk := t.Shape[axis] * inner
		for o := 0; o < outer; o++ {
			copyElems(out, (o*shape[axis]+off)*inner, t, o*chunk, chunk)
		}
		off += t.Shape[axis]
	}
	env[n.Out[0]] = out
	return nil
}

func splitNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	axis := n.intAttr("axis", 0)
	if axis < 0 {
		axis += len(x.Shape)
	}
	if axis < 0 || axis >= len(x.Shape) {
		return fmt.Errorf("bad split axis %d", axis)
	}
	var sizes []int
	if len(n.In) > 1 && n.In[1] != "" {
		st := env[n.In[1]]
		if st.Kind != I64 {
			return fmt.Errorf("split sizes must be int64")
		}
		sizes = make([]int, len(st.I))
		for i, v := range st.I {
			sizes[i] = int(v)
		}
	} else if s := n.intsAttr("split"); s != nil {
		sizes = s
	} else {
		parts := len(n.Out)
		if np := n.intAttr("num_outputs", 0); np > 0 {
			parts = np
		}
		if parts == 0 || x.Shape[axis]%parts != 0 {
			return fmt.Errorf("cannot split axis %d of %d into %d", axis, x.Shape[axis], parts)
		}
		sizes = make([]int, parts)
		for i := range sizes {
			sizes[i] = x.Shape[axis] / parts
		}
	}
	if len(sizes) != len(n.Out) {
		return fmt.Errorf("%d split sizes for %d outputs", len(sizes), len(n.Out))
	}
	sum := 0
	for _, s := range sizes {
		sum += s
	}
	if sum != x.Shape[axis] {
		return fmt.Errorf("split sizes sum to %d, axis has %d", sum, x.Shape[axis])
	}
	inner := numel(x.Shape[axis+1:])
	outer := numel(x.Shape[:axis])
	for oi, size := range sizes {
		shape := append([]int(nil), x.Shape...)
		shape[axis] = size
		out := newTensor(x.Kind, shape)
		start := 0
		for k := 0; k < oi; k++ {
			start += sizes[k]
		}
		for o := 0; o < outer; o++ {
			copyElems(out, o*size*inner, x, (o*x.Shape[axis]+start)*inner, size*inner)
		}
		env[n.Out[oi]] = out
	}
	return nil
}

func gatherNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	idx := env[n.In[1]]
	axis := n.intAttr("axis", 0)
	if axis < 0 {
		axis += len(x.Shape)
	}
	if axis < 0 || axis >= len(x.Shape) {
		return fmt.Errorf("bad gather axis %d", axis)
	}
	if idx.Kind != I64 {
		return fmt.Errorf("indices must be int64")
	}
	d := x.Shape[axis]
	outShape := make([]int, 0, len(x.Shape)-1+len(idx.Shape))
	outShape = append(outShape, x.Shape[:axis]...)
	outShape = append(outShape, idx.Shape...)
	outShape = append(outShape, x.Shape[axis+1:]...)
	out := newTensor(x.Kind, outShape)
	inner := numel(x.Shape[axis+1:])
	outer := numel(x.Shape[:axis])
	for o := 0; o < outer; o++ {
		for j, iv := range idx.I {
			k := int(iv)
			if k < 0 {
				k += d
			}
			if k < 0 || k >= d {
				return fmt.Errorf("index %d out of range for axis size %d", iv, d)
			}
			copyElems(out, (o*len(idx.I)+j)*inner, x, (o*d+k)*inner, inner)
		}
	}
	env[n.Out[0]] = out
	return nil
}

func gatherElementsNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	idx := env[n.In[1]]
	axis := n.intAttr("axis", 0)
	if axis < 0 {
		axis += len(x.Shape)
	}
	if len(idx.Shape) != len(x.Shape) {
		return fmt.Errorf("indices rank %d != data rank %d", len(idx.Shape), len(x.Shape))
	}
	out := newTensor(x.Kind, idx.Shape)
	rank := len(x.Shape)
	xs := strides(x.Shape)
	pos := make([]int, rank)
	d := x.Shape[axis]
	for i := 0; i < out.Numel(); i++ {
		k := int(idx.I[i])
		if k < 0 {
			k += d
		}
		if k < 0 || k >= d {
			return fmt.Errorf("index %d out of range for axis size %d", k, d)
		}
		off := 0
		for r := 0; r < rank; r++ {
			off += pos[r] * xs[r]
		}
		off += (k - pos[axis]) * xs[axis]
		out.set(i, x.at(off))
		for r := rank - 1; r >= 0; r-- {
			pos[r]++
			if pos[r] < idx.Shape[r] {
				break
			}
			pos[r] = 0
		}
	}
	env[n.Out[0]] = out
	return nil
}

func sliceNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	getI := func(i int) *Tensor {
		if i < len(n.In) && n.In[i] != "" {
			return env[n.In[i]]
		}
		return nil
	}
	starts, ends := getI(1), getI(2)
	axesT, stepsT := getI(3), getI(4)
	rank := len(x.Shape)
	axes := make([]int, starts.Numel())
	for i := range axes {
		axes[i] = i
	}
	if axesT != nil {
		for i := range axes {
			axes[i] = int(axesT.I[i])
			if axes[i] < 0 {
				axes[i] += rank
			}
		}
	}
	start := make([]int, rank)
	end := make([]int, rank)
	step := make([]int, rank)
	for i := 0; i < rank; i++ {
		start[i] = 0
		end[i] = x.Shape[i]
		step[i] = 1
	}
	for i, ax := range axes {
		if ax < 0 || ax >= rank {
			return fmt.Errorf("slice axis %d out of range", ax)
		}
		st := int(starts.I[i])
		en := int(ends.I[i])
		sp := 1
		if stepsT != nil {
			sp = int(stepsT.I[i])
		}
		if sp == 0 {
			return fmt.Errorf("slice step 0")
		}
		if sp > 0 {
			if st < 0 {
				st += x.Shape[ax]
			}
			if en < 0 {
				en += x.Shape[ax]
			}
			st = clamp(st, 0, x.Shape[ax])
			en = clamp(en, 0, x.Shape[ax])
		} else {
			if st < 0 {
				st += x.Shape[ax]
			}
			if en < 0 {
				en += x.Shape[ax]
			}
			if st < 0 {
				st = -1
			}
			if st >= x.Shape[ax] {
				st = x.Shape[ax] - 1
			}
			if en < -1 {
				en = -1
			}
			if en >= x.Shape[ax] {
				en = x.Shape[ax] - 1
			}
		}
		start[ax], end[ax], step[ax] = st, en, sp
	}
	outShape := make([]int, rank)
	for i := range outShape {
		if step[i] > 0 {
			d := end[i] - start[i]
			if d < 0 {
				d = 0
			}
			outShape[i] = (d + step[i] - 1) / step[i]
		} else {
			d := start[i] - end[i]
			if d < 0 {
				d = 0
			}
			outShape[i] = (d + (-step[i]) - 1) / (-step[i])
		}
	}
	out := newTensor(x.Kind, outShape)
	xs := strides(x.Shape)
	pos := make([]int, rank)
	for i := 0; i < out.Numel(); i++ {
		off := 0
		for r := 0; r < rank; r++ {
			off += (start[r] + pos[r]*step[r]) * xs[r]
		}
		out.set(i, x.at(off))
		for r := rank - 1; r >= 0; r-- {
			pos[r]++
			if pos[r] < outShape[r] {
				break
			}
			pos[r] = 0
		}
	}
	env[n.Out[0]] = out
	return nil
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func expandNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	st := env[n.In[1]]
	if st.Kind != I64 {
		return fmt.Errorf("expand shape must be int64")
	}
	target := make([]int, len(st.I))
	for i, v := range st.I {
		target[i] = int(v)
	}
	shape, err := broadcastShape(x.Shape, target)
	if err != nil {
		return err
	}
	if len(shape) != len(target) {
		return fmt.Errorf("expand %v to %v", x.Shape, target)
	}
	out := newTensor(x.Kind, shape)
	xs := broadcastStrides(x.Shape, shape)
	pos := make([]int, len(shape))
	for i := 0; i < out.Numel(); i++ {
		off := 0
		for d := range shape {
			off += pos[d] * xs[d]
		}
		out.set(i, x.at(off))
		for d := len(shape) - 1; d >= 0; d-- {
			pos[d]++
			if pos[d] < shape[d] {
				break
			}
			pos[d] = 0
		}
	}
	env[n.Out[0]] = out
	return nil
}

func rangeNode(n *Node, env map[string]*Tensor) error {
	start := env[n.In[0]]
	limit := env[n.In[1]]
	delta := env[n.In[2]]
	if start.Kind == I64 {
		s, _ := scalarI64(start)
		l, _ := scalarI64(limit)
		d, _ := scalarI64(delta)
		if d == 0 {
			return fmt.Errorf("range delta is zero")
		}
		cnt := 0
		if (d > 0 && l > s) || (d < 0 && l < s) {
			cnt = int((l - s + d - sign(d)) / d)
		}
		out := newI64(cnt)
		for i := 0; i < cnt; i++ {
			out.I[i] = s + int64(i)*d
		}
		env[n.Out[0]] = out
		return nil
	}
	s, _ := scalarF32(start)
	l, _ := scalarF32(limit)
	d, _ := scalarF32(delta)
	if d == 0 {
		return fmt.Errorf("range delta is zero")
	}
	cnt := 0
	if (d > 0 && l > s) || (d < 0 && l < s) {
		cnt = int(math.Ceil(float64((l - s) / d)))
		if cnt < 0 {
			cnt = 0
		}
	}
	out := newF32(cnt)
	for i := 0; i < cnt; i++ {
		out.F[i] = s + float32(i)*d
	}
	env[n.Out[0]] = out
	return nil
}

func sign(v int64) int64 {
	if v < 0 {
		return -1
	}
	return 1
}

func constantOfShapeNode(n *Node, env map[string]*Tensor) error {
	st := env[n.In[0]]
	if st.Kind != I64 {
		return fmt.Errorf("shape must be int64")
	}
	shape := make([]int, len(st.I))
	for i, v := range st.I {
		shape[i] = int(v)
	}
	vt, err := n.tensorAttr("value")
	if err != nil {
		return err
	}
	if vt == nil {
		vt = &Tensor{Kind: F32, Shape: []int{1}, F: []float32{0}}
	}
	out := newTensor(vt.Kind, shape)
	for i := 0; i < out.Numel(); i++ {
		out.set(i, vt.at(0))
	}
	env[n.Out[0]] = out
	return nil
}

type reduceOp int

const (
	reduceSum reduceOp = iota
	reduceProd
	reduceMean
)

func reduceNode(n *Node, env map[string]*Tensor, op reduceOp) error {
	x := env[n.In[0]]
	rank := len(x.Shape)
	var axes []int
	if len(n.In) > 1 && n.In[1] != "" {
		at := env[n.In[1]]
		for _, v := range at.I {
			a := int(v)
			if a < 0 {
				a += rank
			}
			axes = append(axes, a)
		}
	} else if a := n.intsAttr("axes"); a != nil {
		for _, v := range a {
			if v < 0 {
				v += rank
			}
			axes = append(axes, v)
		}
	}
	if axes == nil {
		if n.intAttr("noop_with_empty_axes", 0) != 0 {
			env[n.Out[0]] = x
			return nil
		}
		for i := range x.Shape {
			axes = append(axes, i)
		}
	}
	keep := n.intAttr("keepdims", 1) != 0
	red := make(map[int]bool)
	for _, a := range axes {
		red[a] = true
	}
	outShape := make([]int, 0, rank)
	for i, d := range x.Shape {
		if red[i] {
			if keep {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, d)
		}
	}
	out := newTensor(x.Kind, outShape)
	if op == reduceProd {
		for i := 0; i < out.Numel(); i++ {
			out.set(i, 1)
		}
	}
	os := strides(outShape)
	pos := make([]int, rank)
	for i := 0; i < x.Numel(); i++ {
		oi := 0
		j := 0
		for d := 0; d < rank; d++ {
			if red[d] {
				if keep {
					j++
				}
			} else {
				oi += pos[d] * os[j]
				j++
			}
		}
		switch op {
		case reduceProd:
			out.set(oi, out.at(oi)*x.at(i))
		default:
			out.set(oi, out.at(oi)+x.at(i))
		}
		for d := rank - 1; d >= 0; d-- {
			pos[d]++
			if pos[d] < x.Shape[d] {
				break
			}
			pos[d] = 0
		}
	}
	if op == reduceMean {
		cnt := x.Numel() / out.Numel()
		for i := 0; i < out.Numel(); i++ {
			out.set(i, out.at(i)/float64(cnt))
		}
	}
	env[n.Out[0]] = out
	return nil
}

func cumsumNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	axisT := env[n.In[1]]
	ax64, err := scalarI64(axisT)
	if err != nil {
		return err
	}
	axis := int(ax64)
	if axis < 0 {
		axis += len(x.Shape)
	}
	if axis < 0 || axis >= len(x.Shape) {
		return fmt.Errorf("bad cumsum axis %d", axis)
	}
	exclusive := n.intAttr("exclusive", 0) != 0
	reverse := n.intAttr("reverse", 0) != 0
	out := newTensor(x.Kind, x.Shape)
	inner := numel(x.Shape[axis+1:])
	outer := numel(x.Shape[:axis])
	d := x.Shape[axis]
	switch x.Kind {
	case F32:
		for o := 0; o < outer; o++ {
			for in := 0; in < inner; in++ {
				base := o*d*inner + in
				var acc float32
				for k := 0; k < d; k++ {
					idx := k
					if reverse {
						idx = d - 1 - k
					}
					p := base + idx*inner
					if exclusive {
						out.F[p] = acc
						acc += x.F[p]
					} else {
						acc += x.F[p]
						out.F[p] = acc
					}
				}
			}
		}
	case F64:
		for o := 0; o < outer; o++ {
			for in := 0; in < inner; in++ {
				base := o*d*inner + in
				var acc float64
				for k := 0; k < d; k++ {
					idx := k
					if reverse {
						idx = d - 1 - k
					}
					p := base + idx*inner
					if exclusive {
						out.F64[p] = acc
						acc += x.F64[p]
					} else {
						acc += x.F64[p]
						out.F64[p] = acc
					}
				}
			}
		}
	default:
		for o := 0; o < outer; o++ {
			for in := 0; in < inner; in++ {
				base := o*d*inner + in
				var acc int64
				for k := 0; k < d; k++ {
					idx := k
					if reverse {
						idx = d - 1 - k
					}
					p := base + idx*inner
					if exclusive {
						out.I[p] = acc
						acc += x.I[p]
					} else {
						acc += x.I[p]
						out.I[p] = acc
					}
				}
			}
		}
	}
	env[n.Out[0]] = out
	return nil
}
