package engine

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// exec runs one node. env holds every live tensor by name.
func (g *Graph) exec(n *Node, env map[string]*Tensor, rng *rand.Rand) error {
	get := func(i int) (*Tensor, error) {
		if i >= len(n.In) || n.In[i] == "" {
			return nil, nil
		}
		t, ok := env[n.In[i]]
		if !ok {
			return nil, fmt.Errorf("input %q is not available", n.In[i])
		}
		return t, nil
	}
	t0 := func() (*Tensor, error) {
		t, err := get(0)
		if err != nil {
			return nil, err
		}
		if t == nil {
			return nil, fmt.Errorf("missing input 0")
		}
		return t, nil
	}
	put := func(t *Tensor) error {
		if len(n.Out) != 1 {
			return fmt.Errorf("expected 1 output, got %d", len(n.Out))
		}
		env[n.Out[0]] = t
		return nil
	}

	switch n.Op {
	case "Add", "Sub", "Mul", "Div":
		a, err := get(0)
		if err != nil {
			return err
		}
		b, err := get(1)
		if err != nil {
			return err
		}
		var fn func(float64, float64) float64
		if a.Kind == I64 {
			switch n.Op {
			case "Add":
				fn = func(x, y float64) float64 { return float64(int64(x) + int64(y)) }
			case "Sub":
				fn = func(x, y float64) float64 { return float64(int64(x) - int64(y)) }
			case "Mul":
				fn = func(x, y float64) float64 { return float64(int64(x) * int64(y)) }
			default:
				fn = func(x, y float64) float64 {
					if int64(y) == 0 {
						return 0
					}
					return float64(int64(x) / int64(y))
				}
			}
		} else {
			switch n.Op {
			case "Add":
				fn = func(x, y float64) float64 { return x + y }
			case "Sub":
				fn = func(x, y float64) float64 { return x - y }
			case "Mul":
				fn = func(x, y float64) float64 { return x * y }
			default:
				fn = func(x, y float64) float64 { return x / y }
			}
		}
		r, err := binOp(a, b, fn)
		if err != nil {
			return err
		}
		return put(r)

	case "Neg":
		t, err := t0()
		if err != nil {
			return err
		}
		out := newTensor(t.Kind, t.Shape)
		for i := 0; i < t.Numel(); i++ {
			out.set(i, -t.at(i))
		}
		return put(out)
	case "Abs":
		t, err := t0()
		if err != nil {
			return err
		}
		out := newTensor(t.Kind, t.Shape)
		for i := 0; i < t.Numel(); i++ {
			out.set(i, math.Abs(t.at(i)))
		}
		return put(out)
	case "Sin":
		return unaryNode(n, env, math.Sin)
	case "Cos":
		return unaryNode(n, env, math.Cos)
	case "Exp":
		return unaryNode(n, env, math.Exp)
	case "Sqrt":
		return unaryNode(n, env, math.Sqrt)
	case "Reciprocal":
		return unaryNode(n, env, func(x float64) float64 { return 1 / x })
	case "Tanh":
		return unaryNode(n, env, math.Tanh)
	case "Erf":
		return unaryNode(n, env, math.Erf)
	case "Sigmoid":
		return unaryNode(n, env, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) })
	case "Floor":
		return unaryNode(n, env, math.Floor)
	case "Round":
		return unaryNode(n, env, math.RoundToEven)
	case "LeakyRelu":
		alpha := n.floatAttr("alpha", 0.01)
		return unaryNode(n, env, func(x float64) float64 {
			if x >= 0 {
				return x
			}
			return alpha * x
		})

	case "Clip":
		t, err := t0()
		if err != nil {
			return err
		}
		lo, hi := math.Inf(-1), math.Inf(1)
		if mn, _ := get(1); mn != nil {
			v, err := scalarF32(mn)
			if err != nil {
				return err
			}
			lo = float64(v)
		}
		if mx, _ := get(2); mx != nil {
			v, err := scalarF32(mx)
			if err != nil {
				return err
			}
			hi = float64(v)
		}
		out := newTensor(t.Kind, t.Shape)
		for i := 0; i < t.Numel(); i++ {
			v := t.at(i)
			if v < lo {
				v = lo
			}
			if v > hi {
				v = hi
			}
			out.set(i, v)
		}
		return put(out)

	case "Equal", "Less", "Greater", "GreaterOrEqual":
		a, err := get(0)
		if err != nil {
			return err
		}
		b, err := get(1)
		if err != nil {
			return err
		}
		var fn func(float64, float64) bool
		switch n.Op {
		case "Equal":
			fn = func(x, y float64) bool { return x == y }
		case "Less":
			fn = func(x, y float64) bool { return x < y }
		case "Greater":
			fn = func(x, y float64) bool { return x > y }
		case "GreaterOrEqual":
			fn = func(x, y float64) bool { return x >= y }
		}
		r, err := compareTensors(a, b, fn)
		if err != nil {
			return err
		}
		return put(r)

	case "Where":
		c, err := get(0)
		if err != nil {
			return err
		}
		x, err := get(1)
		if err != nil {
			return err
		}
		y, err := get(2)
		if err != nil {
			return err
		}
		return put(where(c, x, y))

	case "Cast":
		t, err := t0()
		if err != nil {
			return err
		}
		to := n.intAttr("to", 1)
		out := &Tensor{Shape: append([]int(nil), t.Shape...)}
		switch to {
		case 1, 10: // float32, float16
			out = newTensor(F32, t.Shape)
			for i := 0; i < t.Numel(); i++ {
				out.F[i] = float32(t.at(i))
			}
		case 11: // double
			out = newTensor(F64, t.Shape)
			for i := 0; i < t.Numel(); i++ {
				out.F64[i] = t.at(i)
			}
		case 6, 7, 9: // int32, int64, bool
			out = newTensor(I64, t.Shape)
			for i := 0; i < t.Numel(); i++ {
				out.I[i] = int64(t.at(i))
			}
		default:
			return fmt.Errorf("unsupported Cast to %d", to)
		}
		return put(out)

	case "Shape":
		return shapeNode(g, n, env)
	case "Reshape":
		return reshapeNode(n, env)
	case "Transpose":
		return transposeNode(n, env)
	case "Squeeze":
		return squeezeNode(n, env, true)
	case "Unsqueeze":
		return squeezeNode(n, env, false)
	case "Concat":
		return concatNode(n, env)
	case "Split":
		return splitNode(n, env)
	case "Gather":
		return gatherNode(n, env)
	case "GatherElements":
		return gatherElementsNode(n, env)
	case "Slice":
		return sliceNode(n, env)
	case "Expand":
		return expandNode(n, env)
	case "Range":
		return rangeNode(n, env)
	case "ConstantOfShape":
		return constantOfShapeNode(n, env)
	case "ReduceSum":
		return reduceNode(n, env, reduceSum)
	case "ReduceProd":
		return reduceNode(n, env, reduceProd)
	case "ReduceMean":
		return reduceNode(n, env, reduceMean)
	case "CumSum":
		return cumsumNode(n, env)
	case "MatMul":
		return matmulNode(n, env)
	case "Gemm":
		return gemmNode(n, env)
	case "Conv":
		return convNode(n, env)
	case "ConvTranspose":
		return convTransposeNode(n, env)
	case "InstanceNormalization":
		return instanceNormNode(n, env)
	case "LayerNormalization":
		return layerNormNode(n, env)
	case "Softmax":
		return softmaxNode(n, env)
	case "LSTM":
		return lstmNode(n, env)
	case "Pad":
		return padNode(n, env)
	case "Resize":
		return resizeNode(n, env)
	case "RandomUniformLike":
		return randomUniformLikeNode(n, env, rng, g.noiseOverride)
	}
	return fmt.Errorf("unsupported op %s", n.Op)
}

func unaryNode(n *Node, env map[string]*Tensor, fn func(float64) float64) error {
	t, ok := env[n.In[0]]
	if !ok {
		return fmt.Errorf("input %q is not available", n.In[0])
	}
	out := newTensor(t.Kind, t.Shape)
	switch t.Kind {
	case F32:
		for i, v := range t.F {
			out.F[i] = float32(fn(float64(v)))
		}
	case F64:
		for i, v := range t.F64 {
			out.F64[i] = fn(v)
		}
	default:
		for i, v := range t.I {
			out.I[i] = int64(fn(float64(v)))
		}
	}
	env[n.Out[0]] = out
	return nil
}

func where(c, x, y *Tensor) *Tensor {
	shape, err := broadcastShape(x.Shape, y.Shape)
	if err != nil {
		shape = x.Shape
	}
	if s, err := broadcastShape(shape, c.Shape); err == nil {
		shape = s
	}
	out := newTensor(x.Kind, shape)
	sc := broadcastStrides(c.Shape, shape)
	sx := broadcastStrides(x.Shape, shape)
	sy := broadcastStrides(y.Shape, shape)
	idx := make([]int, len(shape))
	for i := 0; i < out.Numel(); i++ {
		oc, ox, oy := 0, 0, 0
		for d := range shape {
			oc += idx[d] * sc[d]
			ox += idx[d] * sx[d]
			oy += idx[d] * sy[d]
		}
		if c.at(oc) != 0 {
			out.set(i, x.at(ox))
		} else {
			out.set(i, y.at(oy))
		}
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}
