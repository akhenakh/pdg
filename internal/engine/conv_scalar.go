//go:build !goexperiment.simd

package engine

func depthwiseConv1D(out, x, w, bias []float32, N, C, L, Lout, K, padL int) {
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			var b float32
			if bias != nil {
				b = bias[c]
			}
			orow := out[(n*C+c)*Lout : (n*C+c+1)*Lout]
			xrow := x[(n*C+c)*L : (n*C+c+1)*L]
			wrow := w[c*K : (c+1)*K]
			for o := range orow {
				orow[o] = b
			}
			for k := 0; k < K; k++ {
				wk := wrow[k]
				shift := k - padL
				o0, o1 := 0, Lout
				if o0 < -shift {
					o0 = -shift
				}
				if o1 > L-shift {
					o1 = L - shift
				}
				for o := o0; o < o1; o++ {
					orow[o] += xrow[o+shift] * wk
				}
			}
		}
	}
}

func denseConvChannel(out, x, w, bias []float32, n, oc, N, C, L, Lout, M, K, padL int) {
	orow := out[(n*M+oc)*Lout : (n*M+oc+1)*Lout]
	var b float32
	if bias != nil {
		b = bias[oc]
	}
	for o := range orow {
		orow[o] = b
	}
	for ic := 0; ic < C; ic++ {
		xrow := x[(n*C+ic)*L : (n*C+ic+1)*L]
		wrow := w[(oc*C+ic)*K : (oc*C+ic+1)*K]
		for k := 0; k < K; k++ {
			wk := wrow[k]
			shift := k - padL
			o0, o1 := 0, Lout
			if o0 < -shift {
				o0 = -shift
			}
			if o1 > L-shift {
				o1 = L - shift
			}
			for o := o0; o < o1; o++ {
				orow[o] += xrow[o+shift] * wk
			}
		}
	}
}
