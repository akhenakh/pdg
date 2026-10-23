// Command pdn speaks text with Paradee v2g, entirely on the CPU.
//
//	pdn "Hello, world."              -> pdn.wav
//	pdn -o hello.wav "Hello."        -> hello.wav
//	echo "Long text" | pdn -o - > a.wav
//	pdn -play "Say it out loud."
package main

import (
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/akhenak/paradee-go/internal/tts"
	"github.com/akhenak/paradee-go/internal/wav"
)

const version = "0.1.0"

func main() {
	out := flag.String("o", "pdn.wav", "output WAV file (- for stdout)")
	speed := flag.Float64("speed", 1, "speaking rate (2 = twice as fast)")
	seed := flag.Uint64("seed", 0, "noise seed (0 = random)")
	play := flag.Bool("play", false, "play the audio instead of writing a file")
	quiet := flag.Bool("quiet", false, "do not print statistics")
	ver := flag.Bool("version", false, "print version")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "pdn %s - Paradee text to speech\n\nusage: pdn [flags] [text...]\n\nflags:\n", version)
		flag.PrintDefaults()
	}
	flag.Parse()

	if *ver {
		fmt.Println("pdn", version)
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

	start := time.Now()
	s, err := tts.New()
	if err != nil {
		fatal(err)
	}
	if *seed == 0 {
		*seed = rand.Uint64()
	}
	s.SetSeed(*seed)
	audio, err := s.Synthesize(text, float32(*speed))
	if err != nil {
		fatal(err)
	}
	elapsed := time.Since(start)
	seconds := float64(len(audio)) / tts.SampleRate
	if !*quiet {
		fmt.Fprintf(os.Stderr, "pdn: %.1fs of audio in %v (%.0fx realtime)\n",
			seconds, elapsed.Round(time.Millisecond), seconds/elapsed.Seconds())
	}

	if *play {
		if err := playAudio(audio); err != nil {
			fatal(err)
		}
		return
	}
	if *out == "-" {
		if err := wav.Write(os.Stdout, audio, tts.SampleRate); err != nil {
			fatal(err)
		}
		return
	}
	f, err := os.Create(*out)
	if err != nil {
		fatal(err)
	}
	if err := wav.Write(f, audio, tts.SampleRate); err != nil {
		f.Close()
		fatal(err)
	}
	if err := f.Close(); err != nil {
		fatal(err)
	}
	if !*quiet {
		fmt.Fprintf(os.Stderr, "pdn: wrote %s\n", *out)
	}
}

func playAudio(audio []float32) error {
	players := [][]string{
		{"pw-play"}, {"paplay"}, {"aplay", "-q"}, {"ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"},
	}
	var cmd []string
	for _, p := range players {
		if _, err := exec.LookPath(p[0]); err == nil {
			cmd = p
			break
		}
	}
	if cmd == nil {
		return fmt.Errorf("no audio player found (tried pw-play, paplay, aplay, ffplay); write a file instead")
	}
	dir, err := os.MkdirTemp("", "pdn-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "out.wav")
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := wav.Write(f, audio, tts.SampleRate); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	c := exec.Command(cmd[0], append(cmd[1:], path)...)
	c.Stdin, c.Stdout, c.Stderr = os.Stdin, os.Stderr, os.Stderr
	return c.Run()
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "pdn:", err)
	os.Exit(1)
}
