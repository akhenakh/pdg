// Package pdgo is a zero-install, pure-Go text-to-speech engine.
//
// It turns text into 24 kHz mono float32 audio on the CPU, with no Python,
// JavaScript, ONNX Runtime, or GPU dependency. The model and lexicons are
// embedded in the binary.
//
//	e, err := pdgo.New()
//	if err != nil { ... }
//	audio, err := e.Synthesize(context.Background(), "Hello from Go.")
//	if err != nil { ... }
//	f, _ := os.Create("hello.wav")
//	defer f.Close()
//	pdgo.WriteWAV(f, audio)
//
// For playback or streaming, use Stream. An Engine is safe for concurrent
// use; pass WithSeed for reproducible renders.
package pdgo

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"strings"

	"github.com/akhenak/pdgo/internal/tts"
	"github.com/akhenak/pdgo/internal/wav"
)

// SampleRate is the output sample rate in Hz.
const SampleRate = tts.SampleRate

// ErrNoSpeech is returned when the text has nothing to speak.
var ErrNoSpeech = errors.New("pdgo: no speakable text")

// Engine is a loaded text-to-speech engine. It is safe for concurrent use.
type Engine struct {
	synth *tts.Synth
}

// New loads the embedded model and lexicons. It takes a few milliseconds and
// needs no external files.
func New() (*Engine, error) {
	s, err := tts.New()
	if err != nil {
		return nil, fmt.Errorf("pdgo: %w", err)
	}
	return &Engine{synth: s}, nil
}

// Option configures one Synthesize or Stream call.
type Option func(*config)

type config struct {
	speed  float32
	seed   uint64
	seeded bool
	first  int
	chunk  int
}

func newConfig(opts []Option) config {
	c := config{
		speed: 1,
		first: tts.DefaultFirstFrames,
		chunk: tts.DefaultChunkFrames,
	}
	for _, o := range opts {
		if o != nil {
			o(&c)
		}
	}
	if c.speed <= 0 {
		c.speed = 1
	}
	if c.first <= 0 {
		c.first = tts.DefaultFirstFrames
	}
	if c.chunk <= 0 {
		c.chunk = tts.DefaultChunkFrames
	}
	if !c.seeded {
		c.seed = rand.Uint64()
	}
	return c
}

// WithSpeed sets the speaking rate: 2 speaks twice as fast, 0.5 half as fast.
func WithSpeed(speed float32) Option {
	return func(c *config) {
		if speed > 0 {
			c.speed = speed
		}
	}
}

// WithSeed fixes the noise seed so repeated renders are reproducible.
func WithSeed(seed uint64) Option {
	return func(c *config) {
		c.seed = seed
		c.seeded = true
	}
}

// WithChunking sets the streaming chunk sizes in 80 fps frames (300 samples
// each). The first chunk is short so playback can start early. Zero values use
// the package defaults (8 and 24).
func WithChunking(firstFrames, chunkFrames int) Option {
	return func(c *config) {
		if firstFrames > 0 {
			c.first = firstFrames
		}
		if chunkFrames > 0 {
			c.chunk = chunkFrames
		}
	}
}

// Synthesize returns the complete waveform for text. Samples are mono float32
// at SampleRate, roughly in [-1, 1].
//
// Cancellation is checked between audio chunks; the per-sentence prefix is not
// interruptible.
func (e *Engine) Synthesize(ctx context.Context, text string, opts ...Option) ([]float32, error) {
	if strings.TrimSpace(text) == "" {
		return nil, ErrNoSpeech
	}
	cfg := newConfig(opts)
	var out []float32
	err := e.synth.StreamSeed(text, cfg.speed, cfg.first, cfg.chunk, cfg.seed, func(chunk []float32) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if len(out) == 0 {
			out = make([]float32, 0, len(chunk)*8)
		}
		out = append(out, chunk...)
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, ErrNoSpeech
	}
	return out, nil
}

// Stream synthesizes text and calls yield with each audio chunk as soon as it
// is ready. The first chunk is short (0.1 s by default) so playback can start
// immediately; later chunks are larger. Returning an error from yield stops
// synthesis and returns that error.
//
// Cancellation is checked between chunks; the per-sentence prefix is not
// interruptible.
func (e *Engine) Stream(ctx context.Context, text string, yield func([]float32) error, opts ...Option) error {
	if yield == nil {
		return errors.New("pdgo: nil yield function")
	}
	if strings.TrimSpace(text) == "" {
		return ErrNoSpeech
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	cfg := newConfig(opts)
	return e.synth.StreamSeed(text, cfg.speed, cfg.first, cfg.chunk, cfg.seed, func(chunk []float32) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		return yield(chunk)
	})
}

// WriteWAV writes mono 16-bit PCM WAV at SampleRate.
func WriteWAV(w io.Writer, samples []float32) error {
	return wav.Write(w, samples, SampleRate)
}

// PCM converts float32 samples to little-endian signed 16-bit PCM, ready for
// an audio device or a raw stream.
func PCM(samples []float32) []byte {
	return wav.PCM(samples)
}
