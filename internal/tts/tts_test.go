package tts

import (
	"math"
	"strings"
	"testing"
)

func TestSplitSentences(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello there. How are you?", []string{"Hello there.", "How are you?"}},
		{"Dr. Smith arrived at 5 p.m. yesterday.", []string{"Dr. Smith arrived at 5 p.m. yesterday."}},
		{"J. R. R. Tolkien wrote it. It sold well.", []string{"J. R. R. Tolkien wrote it.", "It sold well."}},
		{"He said \"Stop!\" She left.", []string{"He said \"Stop!\"", "She left."}},
		{"No terminator", []string{"No terminator"}},
	}
	for _, c := range cases {
		got := SplitSentences(c.in)
		if len(got) != len(c.want) {
			t.Fatalf("%q: got %q, want %q", c.in, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("%q: got %q, want %q", c.in, got, c.want)
			}
		}
	}
}

func TestSplitPhonemes(t *testing.T) {
	ps := strings.Repeat("həlˈO, ", 120)
	parts := SplitPhonemes(ps, 100)
	if len(parts) < 2 {
		t.Fatalf("expected several parts, got %d", len(parts))
	}
	for _, p := range parts {
		if n := len([]rune(p)); n > 100 {
			t.Fatalf("part of %d runes exceeds the limit", n)
		}
	}
}

func TestSynthesize(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the model")
	}
	s, err := New()
	if err != nil {
		t.Fatal(err)
	}
	s.SetSeed(7)
	audio, err := s.Synthesize("Hello from Paradee.", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(audio) < 24000/4 {
		t.Fatalf("suspiciously short audio: %d samples", len(audio))
	}
	var peak float64
	for _, v := range audio {
		peak = math.Max(peak, math.Abs(float64(v)))
	}
	if peak < 0.01 || peak > 1.0 {
		t.Fatalf("peak %f out of range", peak)
	}
}

func TestStreamMatchesSynthesize(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the model")
	}
	s, err := New()
	if err != nil {
		t.Fatal(err)
	}
	text := "Hello from Paradee. This is streamed audio."
	s.SetSeed(11)
	want, err := s.Synthesize(text, 1)
	if err != nil {
		t.Fatal(err)
	}
	s.SetSeed(11)
	var got []float32
	err = s.Stream(text, 1, 8, 24, func(chunk []float32) error {
		got = append(got, chunk...)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("streamed %d samples, synthesized %d", len(got), len(want))
	}
	var maxDiff float32
	for i := range got {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	// SIMD FMA rounds differently from the scalar build; the shapes and
	// content must still match.
	if maxDiff > 2e-6 {
		t.Fatalf("streamed vs synthesized max diff %g", maxDiff)
	}
}
