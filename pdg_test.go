package pdg_test

import (
	"bytes"
	"context"
	"errors"
	"math"
	"testing"

	"github.com/akhenak/pdg"
)

func TestSynthesize(t *testing.T) {
	e, err := pdg.New()
	if err != nil {
		t.Fatal(err)
	}
	audio, err := e.Synthesize(context.Background(), "Hello from the pdg library.", pdg.WithSeed(7))
	if err != nil {
		t.Fatal(err)
	}
	if len(audio) < pdg.SampleRate/4 {
		t.Fatalf("too few samples: %d", len(audio))
	}
	var peak float64
	for _, v := range audio {
		peak = math.Max(peak, math.Abs(float64(v)))
	}
	if peak < 0.01 || peak > 1 {
		t.Fatalf("peak %g out of range", peak)
	}
}

func TestStreamMatchesSynthesize(t *testing.T) {
	e, err := pdg.New()
	if err != nil {
		t.Fatal(err)
	}
	text := "Hello from the pdg library. Streaming should match."
	want, err := e.Synthesize(context.Background(), text, pdg.WithSeed(11))
	if err != nil {
		t.Fatal(err)
	}
	var got []float32
	err = e.Stream(context.Background(), text, func(chunk []float32) error {
		got = append(got, chunk...)
		return nil
	}, pdg.WithSeed(11), pdg.WithChunking(4, 8))
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
	if maxDiff > 2e-6 {
		t.Fatalf("streamed vs synthesized max diff %g", maxDiff)
	}
}

func TestWriteWAV(t *testing.T) {
	e, err := pdg.New()
	if err != nil {
		t.Fatal(err)
	}
	audio, err := e.Synthesize(context.Background(), "Wave file.", pdg.WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := pdg.WriteWAV(&buf, audio); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	if len(b) != 44+len(audio)*2 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		t.Fatalf("bad WAV: %d bytes", len(b))
	}
}

func TestEmptyAndCancelled(t *testing.T) {
	e, err := pdg.New()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.Synthesize(context.Background(), "   "); !errors.Is(err, pdg.ErrNoSpeech) {
		t.Fatalf("empty text: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := e.Synthesize(ctx, "Hello"); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled: %v", err)
	}
}
