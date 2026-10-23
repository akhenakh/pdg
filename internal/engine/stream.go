package engine

import (
	"fmt"
	"math/rand/v2"
)

// StreamState is a prepared synthesis: the sentence-global part of the graph
// (text encoder, durations, trunk and noise) has already run, and the
// per-subframe decoder head is ready to produce audio in chunks.
type StreamState struct {
	g        *Graph
	env      map[string]*Tensor
	boundary []string
	N        int // subframes in the head grid
	Total    int // playable samples
}

// Prepare runs the sentence-global prefix of the graph. Each chunk then costs
// only the decoder head.
func (g *Graph) Prepare(inputs map[string]*Tensor, seed uint64) (*StreamState, error) {
	if g.Split == 0 || g.StreamEnd == 0 {
		return nil, fmt.Errorf("model has no streaming split")
	}
	produced := make(map[string]bool)
	for i := g.Split; i < g.StreamEnd; i++ {
		for _, o := range g.Nodes[i].Out {
			produced[o] = true
		}
	}
	need := make(map[string]bool)
	for i := g.Split; i < g.StreamEnd; i++ {
		for _, in := range g.Nodes[i].In {
			if in == "" || produced[in] {
				continue
			}
			if _, isInit := g.init[in]; isInit {
				continue
			}
			need[in] = true
		}
	}

	env := make(map[string]*Tensor, len(g.init)+32)
	for k, v := range g.init {
		env[k] = v
	}
	for k, v := range inputs {
		env[k] = v
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	if err := g.runRange(env, 0, g.Split, rng, need); err != nil {
		return nil, err
	}
	boundary := make([]string, 0, len(need))
	n := 0
	for name := range need {
		t := env[name]
		if t == nil {
			return nil, fmt.Errorf("streaming boundary %q was not produced", name)
		}
		if len(t.Shape) == 0 {
			continue
		}
		d := t.Shape[len(t.Shape)-1]
		if n == 0 {
			n = d
		} else if d != n {
			return nil, fmt.Errorf("streaming boundary %q has length %d, expected %d", name, d, n)
		}
		boundary = append(boundary, name)
	}
	if n < 2 {
		return nil, fmt.Errorf("streaming grid is empty")
	}
	return &StreamState{
		g:        g,
		env:      env,
		boundary: boundary,
		N:        n,
		Total:    (n - 1) * g.SubframeSamples,
	}, nil
}

// Chunk renders output subframes [a,b) and returns their samples. a and b
// should be frame-aligned (even) for 300-sample chunks.
func (s *StreamState) Chunk(a, b int) ([]float32, error) {
	n := s.N - 1
	if a < 0 {
		a = 0
	}
	if b > n {
		b = n
	}
	if a >= b {
		return nil, nil
	}
	// The head's overlap-add needs grid subframe a..b; its local 0 is the
	// incomplete left edge, so render [a, b+1) and skip local subframe 0.
	start := a
	end := b + 1
	if end > s.N {
		end = s.N
	}
	env := make(map[string]*Tensor, len(s.g.init)+len(s.boundary)+8)
	for k, v := range s.g.init {
		env[k] = v
	}
	for _, name := range s.boundary {
		env[name] = sliceLast(s.env[name], start, end)
	}
	rng := rand.New(rand.NewPCG(0, 0))
	if err := s.g.runRange(env, s.g.Split, s.g.StreamEnd, rng, nil); err != nil {
		return nil, err
	}
	yf := env[s.g.streamOut]
	if yf == nil {
		return nil, fmt.Errorf("streaming decoder produced no output")
	}
	off := s.g.SubframeSamples
	cnt := (b - a) * s.g.SubframeSamples
	if off+cnt > len(yf.F) {
		return nil, fmt.Errorf("streaming chunk [%d,%d): output too short (%d < %d)", a, b, len(yf.F), off+cnt)
	}
	out := make([]float32, cnt)
	copy(out, yf.F[off:off+cnt])
	return out, nil
}

// Close releases the prepared prefix.
func (s *StreamState) Close() {
	s.env = nil
	s.boundary = nil
}

// sliceLast copies t[..., a:b] into a new tensor.
func sliceLast(t *Tensor, a, b int) *Tensor {
	rank := len(t.Shape)
	n := t.Shape[rank-1]
	if a < 0 {
		a = 0
	}
	if b > n {
		b = n
	}
	if b < a {
		b = a
	}
	shape := append([]int(nil), t.Shape...)
	shape[rank-1] = b - a
	out := newTensor(t.Kind, shape)
	outer := t.Numel() / n
	for i := 0; i < outer; i++ {
		copyElems(out, i*(b-a), t, i*n+a, b-a)
	}
	return out
}
