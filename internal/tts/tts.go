// Package tts turns text into 24 kHz audio with Paradee v2g.
package tts

import (
	"fmt"
	"regexp"
	"strings"
	"sync/atomic"
	"unicode"

	"github.com/akhenak/pdg/internal/assets"
	"github.com/akhenak/pdg/internal/engine"
	"github.com/akhenak/pdg/internal/phonemize"
)

// SampleRate is Paradee's output rate.
const SampleRate = 24000

// maxPhonemes leaves room for the two padding tokens inside the model's
// 512-token window.
const maxPhonemes = 500

// Default streaming chunk sizes, in 80 fps frames (300 samples each). The
// first chunk is deliberately short so playback can start early.
const (
	DefaultFirstFrames = 8
	DefaultChunkFrames = 24
)

// Synth is a ready-to-use text-to-speech engine. It is safe for concurrent
// use.
type Synth struct {
	graph *engine.Graph
	g2p   *phonemize.Engine
	seed  atomic.Uint64
}

// New loads the embedded model and lexicons.
func New() (*Synth, error) {
	g, err := engine.Load(assets.Model)
	if err != nil {
		return nil, err
	}
	p, err := phonemize.New()
	if err != nil {
		return nil, err
	}
	s := &Synth{graph: g, g2p: p}
	s.seed.Store(1)
	return s, nil
}

// SetSeed fixes the noise seed for reproducible output.
func (s *Synth) SetSeed(seed uint64) { s.seed.Store(seed) }

// Synthesize returns the waveform for text. speed > 1 speaks faster.
func (s *Synth) Synthesize(text string, speed float32) ([]float32, error) {
	return s.SynthesizeSeed(text, speed, s.seed.Add(1))
}

// SynthesizeSeed is Synthesize with an explicit noise seed, so callers can
// reproduce a render without sharing mutable state.
func (s *Synth) SynthesizeSeed(text string, speed float32, seed uint64) ([]float32, error) {
	if speed <= 0 {
		speed = 1
	}
	var audio []float32
	var pieceIndex uint64
	for _, sentence := range SplitSentences(text) {
		phonemes := s.g2p.Phonemize(sentence)
		for _, piece := range SplitPhonemes(phonemes, maxPhonemes) {
			ids := s.g2p.IDs(piece)
			if len(ids) <= 2 {
				continue
			}
			in := &engine.Tensor{Kind: engine.I64, Shape: []int{1, len(ids)}, I: ids}
			sp := &engine.Tensor{Kind: engine.F32, Shape: []int{1}, F: []float32{speed}}
			out, err := s.graph.Run(map[string]*engine.Tensor{
				"input_ids": in,
				"speed":     sp,
			}, seed+pieceIndex)
			if err != nil {
				return nil, err
			}
			wav := out["waveform"]
			audio = append(audio, wav.F...)
			pieceIndex++
		}
	}
	if len(audio) == 0 {
		return nil, fmt.Errorf("no speakable text")
	}
	return audio, nil
}

var abbrevEnd = regexp.MustCompile(`(?:^|[\s"“(])(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Prof|Rev|Gen|Gov|Sen|Rep|Capt|Lt|Col|Sgt|Adm|Mt|Ft|No|Nos|vs|etc|Inc|Ltd|Co|Corp|Bros|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|[A-Z]|(?:[A-Za-z]\.)+[A-Za-z])\.$`)

// SplitSentences splits text the way the model was trained: after . ! ? …
// followed by a capital, digit, quote or bracket, skipping titles and
// initials ("Dr. Smith", "J. R. R. Tolkien").
func SplitSentences(text string) []string {
	text = strings.Join(strings.Fields(text), " ")
	if text == "" {
		return nil
	}
	rs := []rune(text)
	var parts []string
	start := 0
	for i := 0; i < len(rs); i++ {
		if !strings.ContainsRune(".!?…", rs[i]) {
			continue
		}
		j := i + 1
		for j < len(rs) && strings.ContainsRune("\"”’')\\]", rs[j]) {
			j++
		}
		if j >= len(rs) || !unicode.IsSpace(rs[j]) {
			continue
		}
		k := j
		for k < len(rs) && unicode.IsSpace(rs[k]) {
			k++
		}
		if k >= len(rs) || !startsSentence(rs[k]) {
			continue
		}
		parts = append(parts, string(rs[start:j]))
		start = k
		i = k - 1
	}
	parts = append(parts, string(rs[start:]))

	var out []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if n := len(out); n > 0 && abbrevEnd.MatchString(out[n-1]) {
			out[n-1] += " " + p
		} else {
			out = append(out, p)
		}
	}
	return out
}

func startsSentence(r rune) bool {
	return unicode.IsUpper(r) || unicode.IsDigit(r) || r == '"' || r == '“' || r == '('
}

// SplitPhonemes cuts a phoneme string at clause punctuation, else at a word
// break, so each piece fits the model's token window.
func SplitPhonemes(ps string, max int) []string {
	var out []string
	rs := []rune(ps)
	for len(rs) > max {
		head := rs[:max]
		k := -1
		for _, p := range []string{", ", "; ", ": ", "— ", "… "} {
			if idx := lastIndexRunes(head, []rune(p)); idx > k {
				k = idx
			}
		}
		if k > max/3 {
			k++
		} else {
			k = lastIndexRuneSet(head, ' ')
		}
		if k <= 0 {
			k = len(head)
		}
		out = append(out, strings.TrimSpace(string(rs[:k])))
		rs = []rune(strings.TrimSpace(string(rs[k:])))
	}
	if len(rs) > 0 {
		out = append(out, string(rs))
	}
	return out
}

func lastIndexRunes(hay, needle []rune) int {
	for i := len(hay) - len(needle); i >= 0; i-- {
		match := true
		for j := range needle {
			if hay[i+j] != needle[j] {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}

func lastIndexRuneSet(rs []rune, r rune) int {
	for i := len(rs) - 1; i >= 0; i-- {
		if rs[i] == r {
			return i
		}
	}
	return -1
}

// Stream synthesizes text and calls yield with each chunk as soon as it is
// ready. The first chunk is firstFrames long (80 fps frames; 300 samples
// each), later chunks are chunkFrames long. It returns when the text is
// finished or yield returns an error.
func (s *Synth) Stream(text string, speed float32, firstFrames, chunkFrames int, yield func([]float32) error) error {
	return s.StreamSeed(text, speed, firstFrames, chunkFrames, s.seed.Add(1), yield)
}

// StreamSeed is Stream with an explicit noise seed. The seed is advanced per
// synthesized piece, so a fixed seed is reproducible.
func (s *Synth) StreamSeed(text string, speed float32, firstFrames, chunkFrames int, seed uint64, yield func([]float32) error) error {
	if speed <= 0 {
		speed = 1
	}
	if firstFrames <= 0 {
		firstFrames = DefaultFirstFrames
	}
	if chunkFrames <= 0 {
		chunkFrames = DefaultChunkFrames
	}
	var pieceIndex uint64
	for _, sentence := range SplitSentences(text) {
		phonemes := s.g2p.Phonemize(sentence)
		for _, piece := range SplitPhonemes(phonemes, maxPhonemes) {
			ids := s.g2p.IDs(piece)
			if len(ids) <= 2 {
				continue
			}
			in := map[string]*engine.Tensor{
				"input_ids": {Kind: engine.I64, Shape: []int{1, len(ids)}, I: ids},
				"speed":     {Kind: engine.F32, Shape: []int{1}, F: []float32{speed}},
			}
			st, err := s.graph.Prepare(in, seed+pieceIndex)
			pieceIndex++
			if err != nil {
				return err
			}
			total := st.N - 1
			first := true
			for a := 0; a < total; {
				frames := chunkFrames
				if first {
					frames = firstFrames
					first = false
				}
				b := a + frames*2
				if b > total {
					b = total
				}
				chunk, err := st.Chunk(a, b)
				if err != nil {
					st.Close()
					return err
				}
				if err := yield(chunk); err != nil {
					st.Close()
					return err
				}
				a = b
			}
			st.Close()
		}
	}
	return nil
}
