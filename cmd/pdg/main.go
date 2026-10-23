// Command pdg speaks text with Paradee v2g, entirely on the CPU.
//
//	pdg "Hello, world."              -> pdg.wav
//	pdg -o hello.wav "Hello."        -> hello.wav
//	echo "Long text" | pdg -o - > a.wav
//	pdg -play "Say it out loud."
package main

import (
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/akhenak/pdg/internal/tts"
	"github.com/akhenak/pdg/internal/wav"
)

// version is set by GoReleaser (-X main.version=...); local builds say dev.
var version = "dev"

func main() {
	out := flag.String("o", "pdg.wav", "output WAV file (- for stdout)")
	speed := flag.Float64("speed", 1, "speaking rate (2 = twice as fast)")
	seed := flag.Uint64("seed", 0, "noise seed (0 = random)")
	play := flag.Bool("play", false, "play while synthesizing instead of writing a file")
	first := flag.Int("first", tts.DefaultFirstFrames, "frames in the first audio chunk (80 fps, 300 samples each)")
	chunk := flag.Int("chunk", tts.DefaultChunkFrames, "frames per later audio chunk")
	quiet := flag.Bool("quiet", false, "do not print statistics")
	ver := flag.Bool("version", false, "print version")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "pdg %s - Paradee text to speech\n\nusage: pdg [flags] [text...]\n\nflags:\n", version)
		flag.PrintDefaults()
	}
	flag.Parse()

	if *ver {
		fmt.Println("pdg", version)
		return
	}

	text := strings.Join(flag.Args(), " ")
	if text == "" {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			fatal(err)
		}
		text = string(b)
	}
	text = strings.TrimSpace(text)
	if text == "" {
		flag.Usage()
		os.Exit(2)
	}

	synth, err := tts.New()
	if err != nil {
		fatal(err)
	}
	if *seed == 0 {
		*seed = rand.Uint64()
	}
	synth.SetSeed(*seed)

	var sink func([]float32) error
	var finish func() error
	if *play {
		p, err := startPlayer()
		if err != nil {
			fatal(err)
		}
		sink = func(samples []float32) error {
			_, err := p.stdin.Write(wav.PCM(samples))
			return err
		}
		finish = p.close
	} else if *out == "-" {
		if _, err := os.Stdout.Write(wav.Header(tts.SampleRate, 0xFFFFFFFF)); err != nil {
			fatal(err)
		}
		sink = func(samples []float32) error {
			_, err := os.Stdout.Write(wav.PCM(samples))
			return err
		}
		finish = func() error { return nil }
	} else {
		f, err := os.Create(*out)
		if err != nil {
			fatal(err)
		}
		if _, err := f.Write(wav.Header(tts.SampleRate, 0)); err != nil {
			f.Close()
			fatal(err)
		}
		var bytes int
		sink = func(samples []float32) error {
			b := wav.PCM(samples)
			bytes += len(b)
			_, err := f.Write(b)
			return err
		}
		finish = func() error {
			if _, err := f.Seek(0, io.SeekStart); err != nil {
				f.Close()
				return err
			}
			if _, err := f.Write(wav.Header(tts.SampleRate, bytes)); err != nil {
				f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return err
			}
			if !*quiet {
				fmt.Fprintf(os.Stderr, "pdg: wrote %s\n", *out)
			}
			return nil
		}
	}

	start := time.Now()
	var firstAt time.Duration
	var samples int
	err = synth.Stream(text, float32(*speed), *first, *chunk, func(chunk []float32) error {
		if firstAt == 0 {
			firstAt = time.Since(start)
		}
		samples += len(chunk)
		return sink(chunk)
	})
	if err != nil {
		fatal(err)
	}
	if err := finish(); err != nil {
		fatal(err)
	}
	if !*quiet {
		seconds := float64(samples) / tts.SampleRate
		fmt.Fprintf(os.Stderr, "pdg: %.1fs of audio, first audio after %v, total %v (%.0fx realtime)\n",
			seconds, firstAt.Round(time.Millisecond), time.Since(start).Round(time.Millisecond),
			seconds/time.Since(start).Seconds())
	}
}

type audioPlayer struct {
	cmd   *exec.Cmd
	stdin io.WriteCloser
}

func startPlayer() (*audioPlayer, error) {
	candidates := [][]string{
		{"pw-play", "--raw", "--format=s16", "--rate=24000", "--channels=1", "-"},
		{"pw-cat", "--playback", "--raw", "--format=s16", "--rate=24000", "--channels=1", "-"},
		{"paplay", "--raw", "--format=s16le", "--rate=24000", "--channels=1"},
		{"aplay", "-q", "-f", "S16_LE", "-r", "24000", "-c", "1"},
		{"ffplay", "-f", "s16le", "-ar", "24000", "-ac", "1", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"},
	}
	for _, args := range candidates {
		if _, err := exec.LookPath(args[0]); err != nil {
			continue
		}
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		stdin, err := cmd.StdinPipe()
		if err != nil {
			continue
		}
		if err := cmd.Start(); err != nil {
			continue
		}
		return &audioPlayer{cmd: cmd, stdin: stdin}, nil
	}
	return nil, fmt.Errorf("no audio player found (tried pw-play, pw-cat, paplay, aplay, ffplay); write a file with -o instead")
}

func (p *audioPlayer) close() error {
	p.stdin.Close()
	return p.cmd.Wait()
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "pdg:", err)
	os.Exit(1)
}
