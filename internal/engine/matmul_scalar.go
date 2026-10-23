//go:build !goexperiment.simd

package engine

// matmul2D computes C[M,N] = A[M,K] * B[K,N].
func matmul2D(A, B, C []float32, M, K, N int) {
	doRows := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			c := C[i*N : (i+1)*N]
			for j := range c {
				c[j] = 0
			}
			a := A[i*K : (i+1)*K]
			for k, av := range a {
				if av == 0 {
					continue
				}
				b := B[k*N : (k+1)*N]
				for j, bv := range b {
					c[j] += av * bv
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
