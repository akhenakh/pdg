package engine

import (
	"fmt"
	"math"
)

func matmulNode(n *Node, env map[string]*Tensor) error {
	a := env[n.In[0]]
	b := env[n.In[1]]
	if a.Kind != F32 || b.Kind != F32 {
		return fmt.Errorf("MatMul needs float32 inputs")
	}
	aShape, bShape := a.Shape, b.Shape
	squeezeA, squeezeB := false, false
	if len(aShape) == 1 {
		aShape = []int{1, aShape[0]}
		squeezeA = true
	}
	if len(bShape) == 1 {
		bShape = []int{bShape[0], 1}
		squeezeB = true
	}
	if len(aShape) < 2 || len(bShape) < 2 {
		return fmt.Errorf("MatMul needs rank >= 1")
	}
	M, K := aShape[len(aShape)-2], aShape[len(aShape)-1]
	K2, N := bShape[len(bShape)-2], bShape[len(bShape)-1]
	if K != K2 {
		return fmt.Errorf("MatMul %v x %v: inner dimensions differ", a.Shape, b.Shape)
	}
	batchA := aShape[:len(aShape)-2]
	batchB := bShape[:len(bShape)-2]
	batch, err := broadcastShape(batchA, batchB)
	if err != nil {
		return err
	}
	outShape := append(append([]int(nil), batch...), M, N)
	out := newF32(outShape...)
	sa := batchStrides(aShape, batch)
	sb := batchStrides(bShape, batch)
	nb := numel(batch)
	pos := make([]int, len(batch))
	for bi := 0; bi < nb; bi++ {
		oa, ob := 0, 0
		for d := range batch {
			oa += pos[d] * sa[d]
			ob += pos[d] * sb[d]
		}
		matmul2D(a.F[oa:], b.F[ob:], out.F[bi*M*N:], M, K, N)
		for d := len(batch) - 1; d >= 0; d-- {
			pos[d]++
			if pos[d] < batch[d] {
				break
			}
			pos[d] = 0
		}
	}
	if squeezeA || squeezeB {
		// Drop the dimensions added for the vector cases.
		s := outShape
		if squeezeA {
			s = append(s[:len(s)-2], s[len(s)-1])
		}
		if squeezeB {
			s = s[:len(s)-1]
		}
		out.Shape = s
	}
	env[n.Out[0]] = out
	return nil
}

func gemmNode(n *Node, env map[string]*Tensor) error {
	a := env[n.In[0]]
	b := env[n.In[1]]
	alpha := float32(n.floatAttr("alpha", 1))
	beta := float32(n.floatAttr("beta", 1))
	transA := n.intAttr("transA", 0) != 0
	transB := n.intAttr("transB", 0) != 0
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return fmt.Errorf("Gemm needs 2-D inputs, got %v and %v", a.Shape, b.Shape)
	}
	M, K := a.Shape[0], a.Shape[1]
	if transA {
		M, K = K, M
	}
	K2, N := b.Shape[0], b.Shape[1]
	if transB {
		K2, N = N, K2
	}
	if K != K2 {
		return fmt.Errorf("Gemm %v x %v: inner dimensions differ", a.Shape, b.Shape)
	}
	out := newF32(M, N)
	doRows := func(ilo, ihi int) {
		for i := ilo; i < ihi; i++ {
			for k := 0; k < K; k++ {
				var av float32
				if transA {
					av = a.F[k*M+i]
				} else {
					av = a.F[i*K+k]
				}
				if av == 0 {
					continue
				}
				for j := 0; j < N; j++ {
					var bv float32
					if transB {
						bv = b.F[j*K2+k]
					} else {
						bv = b.F[k*N+j]
					}
					out.F[i*N+j] += av * bv
				}
			}
		}
	}
	if M*K*N >= 200000 {
		parallelFor(M, doRows)
	} else {
		doRows(0, M)
	}
	if alpha != 1 {
		for i := range out.F {
			out.F[i] *= alpha
		}
	}
	if len(n.In) > 2 && n.In[2] != "" {
		c := env[n.In[2]]
		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				// C is broadcast over the output: scalar, [N] or [M,N].
				var cv float64
				switch len(c.Shape) {
				case 0:
					cv = c.at(0)
				case 1:
					if c.Shape[0] == N {
						cv = c.at(j)
					} else {
						cv = c.at(0)
					}
				default:
					cv = c.at(i*N + j)
				}
				out.F[i*N+j] += beta * float32(cv)
			}
		}
	}
	env[n.Out[0]] = out
	return nil
}

func convNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	w := env[n.In[1]]
	if len(x.Shape) != 3 {
		return fmt.Errorf("Conv: only 1-D convolutions are supported, got %v", x.Shape)
	}
	if len(w.Shape) != 3 {
		return fmt.Errorf("Conv: weight shape %v", w.Shape)
	}
	var bias *Tensor
	if len(n.In) > 2 && n.In[2] != "" {
		bias = env[n.In[2]]
	}
	N, C, L := x.Shape[0], x.Shape[1], x.Shape[2]
	M, Cg, K := w.Shape[0], w.Shape[1], w.Shape[2]
	group := n.intAttr("group", 1)
	stride := 1
	if s := n.intsAttr("strides"); len(s) > 0 {
		stride = s[0]
	}
	dilation := 1
	if s := n.intsAttr("dilations"); len(s) > 0 {
		dilation = s[0]
	}
	padL, padR := 0, 0
	if p := n.intsAttr("pads"); len(p) == 2 {
		padL, padR = p[0], p[1]
	}
	if C != Cg*group || M%group != 0 {
		return fmt.Errorf("Conv: C=%d group=%d weight=%v", C, group, w.Shape)
	}
	Lout := (L+padL+padR-dilation*(K-1)-1)/stride + 1
	if Lout < 0 {
		Lout = 0
	}
	out := newF32(N, M, Lout)
	if group == M && group == C && Cg == 1 && stride == 1 && dilation == 1 {
		depthwiseConv1D(out.F, x.F, w.F, func() []float32 {
			if bias != nil {
				return bias.F
			}
			return nil
		}(), N, C, L, Lout, K, padL)
		env[n.Out[0]] = out
		return nil
	}
	if group == 1 && stride == 1 && dilation == 1 {
		var b []float32
		if bias != nil {
			b = bias.F
		}
		work := N * M * C * K * Lout
		doCh := func(lo, hi int) {
			for idx := lo; idx < hi; idx++ {
				denseConvChannel(out.F, x.F, w.F, b, idx/M, idx%M, N, C, L, Lout, M, K, padL)
			}
		}
		if work >= 200000 {
			parallelFor(N*M, doCh)
		} else {
			doCh(0, N*M)
		}
		env[n.Out[0]] = out
		return nil
	}
	Mpg := M / group
	work := M * Lout * Cg * K
	for nn := 0; nn < N; nn++ {
		doOC := func(oclo, ochi int) {
			for oc := oclo; oc < ochi; oc++ {
				g := oc / Mpg
				ic0 := g * Cg
				var bv float32
				if bias != nil {
					bv = bias.F[oc]
				}
				obase := oc * Lout
				for o := 0; o < Lout; o++ {
					sum := bv
					xpos := o*stride - padL
					for ic := 0; ic < Cg; ic++ {
						row := ((nn*C)+ic0+ic)*L + xpos
						wrow := (oc*Cg + ic) * K
						for k := 0; k < K; k++ {
							p := xpos + k*dilation
							if p >= 0 && p < L {
								sum += x.F[row+k*dilation] * w.F[wrow+k]
							}
						}
					}
					out.F[obase+o] = sum
				}
			}
		}
		if work >= 200000 {
			parallelFor(M, doOC)
		} else {
			doOC(0, M)
		}
	}
	env[n.Out[0]] = out
	return nil
}

func convTransposeNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	w := env[n.In[1]]
	if len(x.Shape) != 3 || len(w.Shape) != 3 {
		return fmt.Errorf("ConvTranspose: only 1-D is supported")
	}
	var bias *Tensor
	if len(n.In) > 2 && n.In[2] != "" {
		bias = env[n.In[2]]
	}
	N, Cin, L := x.Shape[0], x.Shape[1], x.Shape[2]
	_, CoutG, K := w.Shape[0], w.Shape[1], w.Shape[2]
	group := n.intAttr("group", 1)
	stride := 1
	if s := n.intsAttr("strides"); len(s) > 0 {
		stride = s[0]
	}
	dilation := 1
	if s := n.intsAttr("dilations"); len(s) > 0 {
		dilation = s[0]
	}
	padL, padR := 0, 0
	if p := n.intsAttr("pads"); len(p) == 2 {
		padL, padR = p[0], p[1]
	}
	opad := 0
	if p := n.intsAttr("output_padding"); len(p) > 0 {
		opad = p[0]
	}
	if Cin%group != 0 {
		return fmt.Errorf("ConvTranspose: Cin=%d group=%d", Cin, group)
	}
	Cout := CoutG * group
	Lout := (L-1)*stride - padL - padR + dilation*(K-1) + opad + 1
	if Lout < 0 {
		return fmt.Errorf("ConvTranspose: negative output length")
	}
	out := newF32(N, Cout, Lout)
	if bias != nil {
		for nn := 0; nn < N; nn++ {
			for oc := 0; oc < Cout; oc++ {
				b := bias.F[oc]
				row := out.F[(nn*Cout+oc)*Lout : (nn*Cout+oc+1)*Lout]
				for i := range row {
					row[i] = b
				}
			}
		}
	}
	Cing := Cin / group
	for nn := 0; nn < N; nn++ {
		for ic := 0; ic < Cin; ic++ {
			g := ic / Cing
			for il := 0; il < L; il++ {
				xv := x.F[(nn*Cin+ic)*L+il]
				if xv == 0 {
					continue
				}
				for k := 0; k < K; k++ {
					o := il*stride - padL + k*dilation
					if o < 0 || o >= Lout {
						continue
					}
					for ocg := 0; ocg < CoutG; ocg++ {
						oc := g*CoutG + ocg
						out.F[(nn*Cout+oc)*Lout+o] += xv * w.F[(ic*CoutG+ocg)*K+k]
					}
				}
			}
		}
	}
	env[n.Out[0]] = out
	return nil
}

func instanceNormNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	scale := env[n.In[1]]
	bias := env[n.In[2]]
	eps := n.floatAttr("epsilon", 1e-5)
	if len(x.Shape) < 3 {
		return fmt.Errorf("InstanceNormalization needs rank >= 3")
	}
	N, C := x.Shape[0], x.Shape[1]
	spatial := numel(x.Shape[2:])
	out := newF32(x.Shape...)
	doNC := func(lo, hi int) {
		for nc := lo; nc < hi; nc++ {
			nn, c := nc/C, nc%C
			base := (nn*C + c) * spatial
			row := x.F[base : base+spatial]
			var mean float64
			for _, v := range row {
				mean += float64(v)
			}
			mean /= float64(spatial)
			var vr float64
			for _, v := range row {
				d := float64(v) - mean
				vr += d * d
			}
			vr /= float64(spatial)
			inv := float32(1 / math.Sqrt(vr+eps))
			s := scale.F[c]
			b := bias.F[c]
			o := out.F[base : base+spatial]
			for i, v := range row {
				o[i] = (v-float32(mean))*inv*s + b
			}
		}
	}
	if N*C*spatial >= 200000 {
		parallelFor(N*C, doNC)
	} else {
		doNC(0, N*C)
	}
	env[n.Out[0]] = out
	return nil
}

func layerNormNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	scale := env[n.In[1]]
	var bias *Tensor
	if len(n.In) > 2 && n.In[2] != "" {
		bias = env[n.In[2]]
	}
	axis := n.intAttr("axis", -1)
	if axis < 0 {
		axis += len(x.Shape)
	}
	if axis < 0 || axis >= len(x.Shape) {
		return fmt.Errorf("bad LayerNormalization axis")
	}
	inner := numel(x.Shape[axis:])
	outer := numel(x.Shape[:axis])
	eps := float32(n.floatAttr("epsilon", 1e-5))
	out := newF32(x.Shape...)
	doRows := func(olo, ohi int) {
		for o := olo; o < ohi; o++ {
			row := x.F[o*inner : (o+1)*inner]
			var mean float64
			for _, v := range row {
				mean += float64(v)
			}
			mean /= float64(inner)
			var vr float64
			for _, v := range row {
				d := float64(v) - mean
				vr += d * d
			}
			vr /= float64(inner)
			inv := float32(1 / math.Sqrt(vr+float64(eps)))
			dst := out.F[o*inner : (o+1)*inner]
			for i, v := range row {
				s := float32(1)
				b := float32(0)
				if i < scale.Numel() {
					s = scale.F[i]
				}
				if bias != nil && i < bias.Numel() {
					b = bias.F[i]
				}
				dst[i] = (v-float32(mean))*inv*s + b
			}
		}
	}
	if outer*inner >= 200000 {
		parallelFor(outer, doRows)
	} else {
		doRows(0, outer)
	}
	env[n.Out[0]] = out
	return nil
}

func softmaxNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]]
	axis := n.intAttr("axis", -1)
	if axis < 0 {
		axis += len(x.Shape)
	}
	if axis < 0 || axis >= len(x.Shape) {
		return fmt.Errorf("bad Softmax axis")
	}
	d := x.Shape[axis]
	inner := numel(x.Shape[axis+1:])
	outer := numel(x.Shape[:axis])
	out := newF32(x.Shape...)
	doOuter := func(olo, ohi int) {
		for o := olo; o < ohi; o++ {
			for in := 0; in < inner; in++ {
				base := o*d*inner + in
				maxv := float32(math.Inf(-1))
				for k := 0; k < d; k++ {
					if v := x.F[base+k*inner]; v > maxv {
						maxv = v
					}
				}
				var sum float32
				for k := 0; k < d; k++ {
					v := float32(math.Exp(float64(x.F[base+k*inner] - maxv)))
					out.F[base+k*inner] = v
					sum += v
				}
				if sum != 0 {
					for k := 0; k < d; k++ {
						out.F[base+k*inner] /= sum
					}
				}
			}
		}
	}
	if outer*d*inner >= 200000 {
		parallelFor(outer, doOuter)
	} else {
		doOuter(0, outer)
	}
	env[n.Out[0]] = out
	return nil
}

// lstmNode implements ONNX LSTM (layout=0, bidirectional, no peepholes).
func lstmNode(n *Node, env map[string]*Tensor) error {
	x := env[n.In[0]] // [T, batch, input]
	w := env[n.In[1]] // [dirs, 4H, input]
	r := env[n.In[2]] // [dirs, 4H, H]
	var b *Tensor
	if len(n.In) > 3 && n.In[3] != "" {
		b = env[n.In[3]]
	}
	hidden := n.intAttr("hidden_size", 0)
	if len(x.Shape) != 3 {
		return fmt.Errorf("LSTM: input must be rank 3, got %v", x.Shape)
	}
	T, batch, input := x.Shape[0], x.Shape[1], x.Shape[2]
	dirs := 1
	if n.stringAttr("direction", "forward") == "bidirectional" {
		dirs = 2
	}
	if hidden == 0 {
		hidden = r.Shape[2]
	}
	var h0, c0 *Tensor
	if len(n.In) > 5 && n.In[5] != "" {
		h0 = env[n.In[5]]
	}
	if len(n.In) > 6 && n.In[6] != "" {
		c0 = env[n.In[6]]
	}
	Y := newF32(T, dirs, batch, hidden)
	Yh := newF32(dirs, batch, hidden)
	Yc := newF32(dirs, batch, hidden)
	for d := 0; d < dirs; d++ {
		reverse := d == 1 && dirs == 2
		H := hidden
		W := w.F[d*4*H*input:]
		R := r.F[d*4*H*H:]
		var B []float32
		if b != nil {
			B = b.F[d*8*H:]
		}
		// state per batch
		state := make([]float32, batch*H*2) // h then c
		for bi := 0; bi < batch; bi++ {
			for j := 0; j < H; j++ {
				if h0 != nil {
					state[bi*H+j] = h0.F[(d*batch+bi)*H+j]
				}
				if c0 != nil {
					state[batch*H+bi*H+j] = c0.F[(d*batch+bi)*H+j]
				}
			}
		}
		gates := make([]float32, 4*H)
		cellTmp := make([]float32, H)
		for step := 0; step < T; step++ {
			t := step
			if reverse {
				t = T - 1 - step
			}
			for bi := 0; bi < batch; bi++ {
				hprev := state[bi*H : (bi+1)*H]
				cprev := state[batch*H+bi*H : batch*H+(bi+1)*H]
				for gi := 0; gi < 4*H; gi++ {
					var sum float32
					if B != nil {
						sum = B[gi] + B[4*H+gi]
					}
					wrow := gi * input
					for k := 0; k < input; k++ {
						sum += x.F[(t*batch+bi)*input+k] * W[wrow+k]
					}
					rrow := gi * H
					for k := 0; k < H; k++ {
						sum += hprev[k] * R[rrow+k]
					}
					gates[gi] = sum
				}
				for j := 0; j < H; j++ {
					i := sigmoidf(gates[j])
					o := sigmoidf(gates[H+j])
					f := sigmoidf(gates[2*H+j])
					g := tanhf(gates[3*H+j])
					c := f*cprev[j] + i*g
					cellTmp[j] = c
					Yh.F[(d*batch+bi)*H+j] = o * tanhf(c)
					Y.F[((t*dirs+d)*batch+bi)*H+j] = Yh.F[(d*batch+bi)*H+j]
				}
				copy(state[bi*H:(bi+1)*H], Yh.F[(d*batch+bi)*H:(d*batch+bi+1)*H])
				copy(state[batch*H+bi*H:batch*H+(bi+1)*H], cellTmp)
			}
		}
		copy(Yc.F[d*batch*H:(d+1)*batch*H], state[batch*H:])
	}
	env[n.Out[0]] = Y
	if len(n.Out) > 1 {
		env[n.Out[1]] = Yh
	}
	if len(n.Out) > 2 {
		env[n.Out[2]] = Yc
	}
	return nil
}

func sigmoidf(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }
func tanhf(x float32) float32    { return float32(math.Tanh(float64(x))) }
