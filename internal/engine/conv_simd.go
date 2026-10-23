//go:build goexperiment.simd

package engine

import "simd"

func depthwiseConv1D(out, x, w, bias []float32, N, C, L, Lout, K, padL int) {
	var zero simd.Float32s
	vl := zero.Len()
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			var b float32
			if bias != nil {
				b = bias[c]
			}
			orow := out[(n*C+c)*Lout : (n*C+c+1)*Lout]
			xrow := x[(n*C+c)*L : (n*C+c+1)*L]
			wrow := w[c*K : (c+1)*K]
			vb := simd.BroadcastFloat32s(b)
			o := 0
			for ; o+vl <= Lout; o += vl {
				vb.Store(orow[o:])
			}
			for ; o < Lout; o++ {
				orow[o] = b
			}
			for k := 0; k < K; k++ {
				wk := wrow[k]
				if wk == 0 {
					continue
				}
				shift := k - padL
				o0, o1 := 0, Lout
				if o0 < -shift {
					o0 = -shift
				}
				if o1 > L-shift {
					o1 = L - shift
				}
				vw := simd.BroadcastFloat32s(wk)
				o = o0
				for ; o+vl <= o1; o += vl {
					vc := simd.LoadFloat32s(orow[o:])
					vx := simd.LoadFloat32s(xrow[o+shift:])
					vc = vx.MulAdd(vw, vc)
					vc.Store(orow[o:])
				}
				for ; o < o1; o++ {
					orow[o] += xrow[o+shift] * wk
				}
			}
		}
	}
}

// denseConvChannel computes one output channel of a group-1, stride-1,
// dilation-1 1-D convolution, vectorized over time.
func denseConvChannel(out, x, w, bias []float32, n, oc, N, C, L, Lout, M, K, padL int) {
	var zero simd.Float32s
	vl := zero.Len()
	orow := out[(n*M+oc)*Lout : (n*M+oc+1)*Lout]
	var b float32
	if bias != nil {
		b = bias[oc]
	}
	vb := simd.BroadcastFloat32s(b)
	o := 0
	for ; o+vl <= Lout; o += vl {
		vb.Store(orow[o:])
	}
	for ; o < Lout; o++ {
		orow[o] = b
	}
	for ic := 0; ic < C; ic++ {
		xrow := x[(n*C+ic)*L : (n*C+ic+1)*L]
		wrow := w[(oc*C+ic)*K : (oc*C+ic+1)*K]
		for k := 0; k < K; k++ {
			wk := wrow[k]
			if wk == 0 {
				continue
			}
			shift := k - padL
			o0, o1 := 0, Lout
			if o0 < -shift {
				o0 = -shift
			}
			if o1 > L-shift {
				o1 = L - shift
			}
			vw := simd.BroadcastFloat32s(wk)
			o = o0
			for ; o+vl <= o1; o += vl {
				vc := simd.LoadFloat32s(orow[o:])
				vx := simd.LoadFloat32s(xrow[o+shift:])
				vc = vx.MulAdd(vw, vc)
				vc.Store(orow[o:])
			}
			for ; o < o1; o++ {
				orow[o] += xrow[o+shift] * wk
			}
		}
	}
}
