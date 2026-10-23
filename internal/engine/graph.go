package engine

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
)

// Node is one instruction of the model.
type Node struct {
	Op    string
	Name  string
	In    []string
	Out   []string
	Attrs map[string]any
}

// Graph is the compiled model. It is immutable and safe for concurrent use
// if every Run gets its own environment (it does).
type Graph struct {
	SampleRate int
	Inputs     []string
	Outputs    []string
	Nodes      []Node
	init       map[string]*Tensor
	lastUse    []map[string]bool // tensors that may be freed after node i

	// Streaming split (v2g): nodes [0,Split) are the sentence-global text
	// side + trunk + noise; nodes [Split,StreamEnd) are the per-subframe
	// decoder head.
	Split           int
	StreamEnd       int
	SubframeSamples int
	streamOut       string

	// noiseOverride replaces RandomUniformLike for deterministic tests.
	noiseOverride *float32

	// trace, when set, is called after every node with the node index and env.
	trace func(i int, n *Node, env map[string]*Tensor)
}

type header struct {
	Format     string `json:"format"`
	Version    int    `json:"version"`
	Source     string `json:"source"`
	SampleRate int    `json:"sample_rate"`
	Inputs     []struct {
		Name  string `json:"name"`
		DType string `json:"dtype"`
		Rank  int    `json:"rank"`
	} `json:"inputs"`
	Outputs []string `json:"outputs"`
	Stream  *struct {
		Split           int `json:"split"`
		End             int `json:"end"`
		SubframeSamples int `json:"subframe_samples"`
	} `json:"stream"`
	Tensors []struct {
		Name  string `json:"name"`
		DType string `json:"dtype"`
		Shape []int  `json:"shape"`
		Off   int    `json:"off"`
		N     int    `json:"n"`
	} `json:"tensors"`
	Nodes []struct {
		Op    string         `json:"op"`
		Name  string         `json:"name"`
		In    []string       `json:"in"`
		Out   []string       `json:"out"`
		Attrs map[string]any `json:"attrs"`
	} `json:"nodes"`
}

// Load parses a model produced by tools/convert.py.
func Load(data []byte) (*Graph, error) {
	if len(data) < 12 || string(data[:4]) != "PDNG" {
		return nil, fmt.Errorf("not a pdg model")
	}
	version := binary.LittleEndian.Uint32(data[4:8])
	if version != 1 {
		return nil, fmt.Errorf("unsupported model version %d", version)
	}
	n := binary.LittleEndian.Uint32(data[8:12])
	if uint64(12)+uint64(n) > uint64(len(data)) {
		return nil, fmt.Errorf("truncated model header")
	}
	var h header
	if err := json.Unmarshal(data[12:12+n], &h); err != nil {
		return nil, fmt.Errorf("model header: %w", err)
	}
	blob := data[12+n:]

	g := &Graph{SampleRate: h.SampleRate, Outputs: h.Outputs, init: make(map[string]*Tensor, len(h.Tensors))}
	for _, in := range h.Inputs {
		g.Inputs = append(g.Inputs, in.Name)
	}
	for _, t := range h.Tensors {
		sz := t.N
		var elem int
		switch t.DType {
		case "f32":
			elem = 4
		case "f64":
			elem = 8
		case "i64":
			elem = 8
		default:
			return nil, fmt.Errorf("tensor %s: unknown dtype %q", t.Name, t.DType)
		}
		if t.Off < 0 || t.Off+sz*elem > len(blob) {
			return nil, fmt.Errorf("tensor %s: out of bounds", t.Name)
		}
		if numel(t.Shape) != sz {
			return nil, fmt.Errorf("tensor %s: shape %v does not match %d elements", t.Name, t.Shape, sz)
		}
		raw := blob[t.Off : t.Off+sz*elem]
		if t.DType == "f32" {
			f := make([]float32, sz)
			for i := range f {
				f[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
			}
			g.init[t.Name] = &Tensor{Kind: F32, Shape: t.Shape, F: f}
		} else if t.DType == "f64" {
			f := make([]float64, sz)
			for i := range f {
				f[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
			}
			g.init[t.Name] = &Tensor{Kind: F64, Shape: t.Shape, F64: f}
		} else {
			v := make([]int64, sz)
			for i := range v {
				v[i] = int64(binary.LittleEndian.Uint64(raw[i*8:]))
			}
			g.init[t.Name] = &Tensor{Kind: I64, Shape: t.Shape, I: v}
		}
	}
	for _, n := range h.Nodes {
		g.Nodes = append(g.Nodes, Node{Op: n.Op, Name: n.Name, In: n.In, Out: n.Out, Attrs: n.Attrs})
	}
	if h.Stream != nil {
		if h.Stream.Split <= 0 || h.Stream.End > len(g.Nodes) || h.Stream.Split >= h.Stream.End {
			return nil, fmt.Errorf("invalid streaming split %d..%d for %d nodes", h.Stream.Split, h.Stream.End, len(g.Nodes))
		}
		g.Split = h.Stream.Split
		g.StreamEnd = h.Stream.End
		g.SubframeSamples = h.Stream.SubframeSamples
		if len(g.Nodes[g.StreamEnd-1].Out) == 0 {
			return nil, fmt.Errorf("streaming end node has no output")
		}
		g.streamOut = g.Nodes[g.StreamEnd-1].Out[0]
	}
	g.computeLastUse()
	return g, nil
}

// LoadFile is Load for a path on disk.
func LoadFile(path string) (*Graph, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Load(data)
}

func (g *Graph) computeLastUse() {
	last := make(map[string]int, len(g.init))
	for i, n := range g.Nodes {
		for _, in := range n.In {
			if in != "" {
				last[in] = i
			}
		}
	}
	for _, o := range g.Outputs {
		last[o] = len(g.Nodes) - 1
	}
	g.lastUse = make([]map[string]bool, len(g.Nodes))
	for name, i := range last {
		if g.lastUse[i] == nil {
			g.lastUse[i] = make(map[string]bool)
		}
		g.lastUse[i][name] = true
	}
}

// Run executes the graph. Inputs must contain every graph input; results are
// returned by output name.
func (g *Graph) Run(inputs map[string]*Tensor, seed uint64) (map[string]*Tensor, error) {
	env := make(map[string]*Tensor, len(g.init)+32)
	for k, v := range g.init {
		env[k] = v
	}
	for k, v := range inputs {
		env[k] = v
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	if err := g.runRange(env, 0, len(g.Nodes), rng, nil); err != nil {
		return nil, err
	}
	out := make(map[string]*Tensor, len(g.Outputs))
	for _, name := range g.Outputs {
		t, ok := env[name]
		if !ok {
			return nil, fmt.Errorf("output %q was not produced", name)
		}
		out[name] = t
	}
	return out, nil
}

// runRange executes nodes [from,to), freeing dead tensors unless they are in
// keep (the streaming boundary) or a graph output.
func (g *Graph) runRange(env map[string]*Tensor, from, to int, rng *rand.Rand, keep map[string]bool) error {
	for i := from; i < to; i++ {
		n := &g.Nodes[i]
		if err := g.exec(n, env, rng); err != nil {
			return fmt.Errorf("node %d %s %q: %w", i, n.Op, n.Name, err)
		}
		if g.trace != nil {
			g.trace(i, n, env)
		}
		if g.lastUse[i] == nil {
			continue
		}
		for name := range g.lastUse[i] {
			if keep[name] {
				continue
			}
			if _, isInit := g.init[name]; isInit {
				continue
			}
			isOutput := false
			for _, o := range g.Outputs {
				if o == name {
					isOutput = true
					break
				}
			}
			if !isOutput {
				delete(env, name)
			}
		}
	}
	return nil
}

// Attr helpers -------------------------------------------------------------

func (n *Node) attr(name string) (any, bool) {
	v, ok := n.Attrs[name]
	return v, ok
}

func (n *Node) intAttr(name string, def int) int {
	if v, ok := n.Attrs[name]; ok {
		if f, ok := v.(float64); ok {
			return int(f)
		}
	}
	return def
}

func (n *Node) floatAttr(name string, def float64) float64 {
	if v, ok := n.Attrs[name]; ok {
		if f, ok := v.(float64); ok {
			return f
		}
	}
	return def
}

func (n *Node) stringAttr(name string, def string) string {
	if v, ok := n.Attrs[name]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return def
}

func (n *Node) intsAttr(name string) []int {
	v, ok := n.Attrs[name]
	if !ok {
		return nil
	}
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]int, len(arr))
	for i, x := range arr {
		switch t := x.(type) {
		case float64:
			out[i] = int(t)
		case int:
			out[i] = t
		}
	}
	return out
}

func (n *Node) tensorAttr(name string) (*Tensor, error) {
	v, ok := n.Attrs[name]
	if !ok {
		return nil, nil
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("attribute %s is not a tensor", name)
	}
	tm, ok := m["tensor"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("attribute %s has no tensor payload", name)
	}
	dtype, _ := tm["dtype"].(string)
	var shape []int
	if ss, ok := tm["shape"].([]any); ok {
		for _, s := range ss {
			if f, ok := s.(float64); ok {
				shape = append(shape, int(f))
			}
		}
	}
	data, _ := tm["data"].([]any)
	t := &Tensor{Shape: shape}
	if dtype == "f32" {
		t.Kind = F32
		t.F = make([]float32, len(data))
		for i, x := range data {
			if f, ok := x.(float64); ok {
				t.F[i] = float32(f)
			}
		}
	} else if dtype == "f64" {
		t.Kind = F64
		t.F64 = make([]float64, len(data))
		for i, x := range data {
			if f, ok := x.(float64); ok {
				t.F64[i] = f
			}
		}
	} else {
		t.Kind = I64
		t.I = make([]int64, len(data))
		for i, x := range data {
			if f, ok := x.(float64); ok {
				t.I[i] = int64(f)
			}
		}
	}
	return t, nil
}
