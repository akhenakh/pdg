package engine

import (
	"fmt"
	"math"
	"math/rand/v2"
)

func padNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	padsT := env[n.In[1]]
	if padsT.Kind != I64 || len(padsT.I) != 2*len(x.Shape) {
		return fmt.Errorf("Pad: pads must be int64 with %d values, got %v", 2*len(x.Shape), padsT.Shape)
	}
	mode := n.stringAttr("mode", "constant")
	var cval float32
	if len(n.In) > 2 && n.In[2] != "" {
		v, err := scalarF32(env[n.In[2]])
		if err != nil {
			return err
		}
		cval = v
	}
	rank := len(x.Shape)
	outShape := make([]int, rank)
	before := make([]int, rank)
	for i := 0; i < rank; i++ {
		before[i] = int(padsT.I[i])
		after := int(padsT.I[rank+i])
		outShape[i] = x.Shape[i] + before[i] + after
		if outShape[i] < 0 {
			return fmt.Errorf("Pad: negative output dimension")
		}
	}
	out := newTensor(x.Kind, outShape)
	xs := strides(x.Shape)
	os := strides(outShape)
	pos := make([]int, rank)
	for i := 0; i < out.Numel(); i++ {
		off := 0
		ok := true
		for d := 0; d < rank; d++ {
			src := pos[d] - before[d]
			switch mode {
			case "constant":
				if src < 0 || src >= x.Shape[d] {
					ok = false
				}
			case "reflect":
				src = reflectIndex(src, x.Shape[d])
			case "edge":
				if src < 0 {
					src = 0
				}
				if src >= x.Shape[d] {
					src = x.Shape[d] - 1
				}
			default:
				return fmt.Errorf("Pad: unsupported mode %q", mode)
			}
			if ok {
				off += src * xs[d]
			}
		}
		if ok {
			out.set(i, x.at(off))
		} else {
			out.set(i, float64(cval))
		}
		for d := rank - 1; d >= 0; d-- {
			pos[d]++
			if pos[d] < outShape[d] {
				break
			}
			pos[d] = 0
		}
	}
	_ = os
	env[n.Out[0]] = out
	return nil
}

func reflectIndex(i, n int) int {
	if n == 1 {
		return 0
	}
	period := 2 * (n - 1)
	i %= period
	if i < 0 {
		i += period
	}
	if i >= n {
		i = period - i
	}
	return i
}

func resizeNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	rank := len(x.Shape)
	var scales []float32
	var sizes []int
	if len(n.In) > 2 && n.In[2] != "" {
		t := env[n.In[2]]
		if t.Numel() != rank {
			return fmt.Errorf("Resize: %d scales for rank %d", t.Numel(), rank)
		}
		scales = make([]float32, rank)
		for i := range scales {
			if t.Kind == F32 {
				scales[i] = t.F[i]
			} else {
				scales[i] = float32(t.I[i])
			}
		}
	} else if len(n.In) > 3 && n.In[3] != "" {
		t := env[n.In[3]]
		if t.Numel() != rank {
			return fmt.Errorf("Resize: %d sizes for rank %d", t.Numel(), rank)
		}
		sizes = make([]int, rank)
		for i := range sizes {
			sizes[i] = int(t.I[i])
		}
	} else {
		return fmt.Errorf("Resize: neither scales nor sizes given")
	}
	mode := n.stringAttr("mode", "nearest")
	if mode != "nearest" {
		return fmt.Errorf("Resize: only nearest is supported, got %q", mode)
	}
	nearestMode := n.stringAttr("nearest_mode", "floor")
	outShape := make([]int, rank)
	for i := 0; i < rank; i++ {
		if sizes != nil {
			outShape[i] = sizes[i]
		} else {
			outShape[i] = int(math.Floor(float64(float32(x.Shape[i]) * scales[i])))
			if outShape[i] < 1 {
				outShape[i] = 1
			}
		}
	}
	out := newTensor(x.Kind, outShape)
	xs := strides(x.Shape)
	pos := make([]int, rank)
	srcIdx := make([]int, rank)
	for i := 0; i < out.Numel(); i++ {
		for d := 0; d < rank; d++ {
			var s float64
			if sizes != nil {
				s = float64(x.Shape[d]) / float64(outShape[d])
			} else {
				s = 1 / float64(scales[d])
			}
			v := float64(pos[d]) * s
			switch nearestMode {
			case "floor":
				srcIdx[d] = int(math.Floor(v))
			case "round_prefer_floor":
				srcIdx[d] = int(math.Floor(v + 0.5))
				if math.Abs(v-math.Floor(v)-0.5) < 1e-9 {
					srcIdx[d] = int(math.Floor(v))
				}
			case "round_prefer_ceil":
				srcIdx[d] = int(math.Floor(v + 0.5))
			case "ceil":
				srcIdx[d] = int(math.Ceil(v))
			default:
				srcIdx[d] = int(math.Floor(v))
			}
			if srcIdx[d] < 0 {
				srcIdx[d] = 0
			}
			if srcIdx[d] >= x.Shape[d] {
				srcIdx[d] = x.Shape[d] - 1
			}
		}
		off := 0
		for d := 0; d < rank; d++ {
			off += srcIdx[d] * xs[d]
		}
		out.set(i, x.at(off))
		for d := rank - 1; d >= 0; d-- {
			pos[d]++
			if pos[d] < outShape[d] {
				break
			}
			pos[d] = 0
		}
	}
	env[n.Out[0]] = out
	return nil
}

func randomUniformLikeNode(n *Node, env map[string]*Tensor, rng *rand.Rand, override *float32) error {
	x := env[n.In[0]]
	low := float32(n.floatAttr("low", 0))
	high := float32(n.floatAttr("high", 1))
	dtype := n.intAttr("dtype", 1)
	shape := append([]int(nil), x.Shape...)
	if dtype == 1 || dtype == 10 || dtype == 11 {
		out := newF32(shape...)
		for i := range out.F {
			if override != nil {
				out.F[i] = *override
			} else {
				out.F[i] = low + (high-low)*rng.Float32()
			}
		}
		env[n.Out[0]] = out
		return nil
	}
	out := newI64(shape...)
	for i := range out.I {
		out.I[i] = int64(low + (high-low)*rng.Float32())
	}
	env[n.Out[0]] = out
	return nil
}
