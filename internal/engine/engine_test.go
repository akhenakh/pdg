package engine

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
	"time"
)

type testCase struct {
	IDs     []int64 `json:"ids"`
	Samples int     `json:"samples"`
	File    string  `json:"file"`
}

func TestReference(t *testing.T) {
	g, err := LoadFile("../assets/paradee.bin")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Cases []testCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	noise := float32(0.5)
	g.noiseOverride = &noise
	for i, c := range doc.Cases {
		ids := &Tensor{Kind: I64, Shape: []int{1, len(c.IDs)}, I: c.IDs}
		speed := &Tensor{Kind: F32, Shape: []int{1}, F: []float32{1}}
		start := time.Now()
		out, err := g.Run(map[string]*Tensor{"input_ids": ids, "speed": speed}, 1)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		dt := time.Since(start)
		y := out["waveform"]
		refRaw, err := os.ReadFile("testdata/" + c.File)
		if err != nil {
			t.Fatal(err)
		}
		if len(refRaw) != len(y.F)*4 {
			t.Fatalf("case %d: got %d samples, reference has %d", i, len(y.F), len(refRaw)/4)
		}
		var maxAbs, sumSq, refSq float64
		worst := -1
		for j := range y.F {
			v := float64(y.F[j])
			r := float64(readF32(refRaw[j*4:]))
			d := math.Abs(v - r)
			if d > maxAbs {
				maxAbs = d
				worst = j
			}
			sumSq += d * d
			refSq += r * r
		}
		rms := math.Sqrt(sumSq / float64(len(y.F)))
		rel := math.Sqrt(sumSq/refSq) * 100
		seconds := float64(len(y.F)) / 24000
		t.Logf("case %d: %d samples (%.1fs audio) in %v (%.0fx realtime), max|diff|=%.3g at %d, rms=%.3g, rel=%.3f%%",
			i, len(y.F), seconds, dt.Round(time.Millisecond), seconds/dt.Seconds(), maxAbs, worst, rms, rel)
		if maxAbs > 2e-3 {
			t.Errorf("case %d: max abs diff %g exceeds tolerance", i, maxAbs)
		}
	}
}

func readF32(b []byte) float32 {
	return math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
}

func BenchmarkSynth(b *testing.B) {
	g, err := LoadFile("../assets/paradee.bin")
	if err != nil {
		b.Fatal(err)
	}
	ids := make([]int64, 80)
	for i := range ids {
		ids[i] = int64(50 + i%100)
	}
	ids[0], ids[len(ids)-1] = 0, 0
	in := map[string]*Tensor{
		"input_ids": {Kind: I64, Shape: []int{1, len(ids)}, I: ids},
		"speed":     {Kind: F32, Shape: []int{1}, F: []float32{1}},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := g.Run(in, uint64(i))
		if err != nil {
			b.Fatal(err)
		}
		if len(out["waveform"].F) == 0 {
			b.Fatal("no audio")
		}
	}
}

var _ = fmt.Sprint
