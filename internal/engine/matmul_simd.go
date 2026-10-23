//go:build goexperiment.simd

package engine

import "simd"

// matmul2D computes C[M,N] = A[M,K] * B[K,N] with a vectorized inner loop.
func matmul2D(A, B, C []float32, M, K, N int) {
	var zero simd.Float32s
	vl := zero.Len()
	doRows := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			c := C[i*N : (i+1)*N]
			j := 0
			for ; j+vl <= N; j += vl {
				zero.Store(c[j:])
			}
			for ; j < N; j++ {
				c[j] = 0
			}
			a := A[i*K : (i+1)*K]
			for k, av := range a {
				if av == 0 {
					continue
				}
				vb := simd.BroadcastFloat32s(av)
				b := B[k*N : (k+1)*N]
				j = 0
				for ; j+vl <= N; j += vl {
					vc := simd.LoadFloat32s(c[j:])
					vx := simd.LoadFloat32s(b[j:])
					vc = vx.MulAdd(vb, vc)
					vc.Store(c[j:])
				}
				for ; j < N; j++ {
					c[j] += av * b[j]
				}
			}
		}
	}
	if M*K*N >= 200000 {
		parallelFor(M, doRows)
	} else {
		doRows(0, M)
	}
}
