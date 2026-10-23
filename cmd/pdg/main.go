// Command pdg speaks text with Paradee v2g, entirely on the CPU.
//
//	pdg "Hello, world."              -> plays on the default device
//	pdg -o hello.wav "Hello."        -> hello.wav
//	echo "Long text" | pdg -o - > a.wav
//	pdg -play=false "Hello."         -> pdg.wav
package main

import (
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/akhenak/pdg/internal/tts"
	"github.com/akhenak/pdg/internal/wav"
)

// version is set by GoReleaser (-X main.version=...); local builds say dev.
var version = "dev"

func main() {
	out := flag.String("o", "", "output WAV file instead of playing (- for stdout)")
	speed := flag.Float64("speed", 1, "speaking rate (2 = twice as fast)")
	seed := flag.Uint64("seed", 0, "noise seed (0 = random)")
	play := flag.Bool("play", true, "play audio (default; ignored when -o is set)")
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
	var start time.Time
	var playAt time.Duration
	bufferedPlay := false
	path := *out
	if !*play && path == "" {
		path = "pdg.wav"
	}
	if *play && path == "" {
		if p, err := startStreamingPlayer(); err == nil {
			// Keep the encoded audio while streaming: if the player dies
			// (unsupported flags, missing audio device, ...), restart
			// playback from the beginning through the buffered player
			// instead of failing with a broken pipe.
			fallback := canPlayBuffered()
			var streamed []byte
			sink = func(samples []float32) error {
				pcm := wav.PCM(samples)
				if fallback {
					streamed = append(streamed, pcm...)
				}
				if p == nil {
					return nil
				}
				if _, err := p.stdin.Write(pcm); err != nil {
					p.abandon()
					p = nil
					if !fallback {
						return err
					}
					bufferedPlay = true
				}
				return nil
			}
			finish = func() error {
				if p == nil {
					playAt = time.Since(start)
					return playBufferedPCM(streamed)
				}
				return p.close()
			}
		} else if canPlayBuffered() {
			// macOS afplay and Windows PowerShell's SoundPlayer need a
			// complete file, so buffer and play once synthesis is done.
			bufferedPlay = true
			var audio []float32
			sink = func(samples []float32) error {
				audio = append(audio, samples...)
				return nil
			}
			finish = func() error {
				// afplay needs the complete file, so playback starts now.
				playAt = time.Since(start)
				return playBuffered(audio)
			}
		} else {
			fatal(fmt.Errorf("no audio player found (tried pw-play, pw-cat, paplay, aplay, ffplay, afplay, PowerShell); write a file with -o instead"))
		}
	} else if path == "-" {
		if _, err := os.Stdout.Write(wav.Header(tts.SampleRate, 0xFFFFFFFF)); err != nil {
			fatal(err)
		}
		sink = func(samples []float32) error {
			_, err := os.Stdout.Write(wav.PCM(samples))
			return err
		}
		finish = func() error { return nil }
	} else {
		f, err := os.Create(path)
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
				fmt.Fprintf(os.Stderr, "pdg: wrote %s\n", path)
			}
			return nil
		}
	}

	start = time.Now()
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
		total := time.Since(start)
		if bufferedPlay {
			if playAt == 0 {
				playAt = total
			}
			fmt.Fprintf(os.Stderr, "pdg: %.1fs of audio, playback started after %v, total %v (%.0fx realtime)\n",
				seconds, playAt.Round(time.Millisecond), total.Round(time.Millisecond),
				seconds/total.Seconds())
		} else {
			fmt.Fprintf(os.Stderr, "pdg: %.1fs of audio, first audio after %v, total %v (%.0fx realtime)\n",
				seconds, firstAt.Round(time.Millisecond), total.Round(time.Millisecond),
				seconds/total.Seconds())
		}
	}
}

type audioPlayer struct {
	cmd   *exec.Cmd
	stdin io.WriteCloser
}

func startStreamingPlayer() (*audioPlayer, error) {
	candidates := [][]string{
		{"pw-play", "--raw", "--format=s16", "--rate=24000", "--channels=1", "-"},
		{"pw-cat", "--playback", "--raw", "--format=s16", "--rate=24000", "--channels=1", "-"},
		{"paplay", "--raw", "--format=s16le", "--rate=24000", "--channels=1"},
		{"aplay", "-q", "-f", "S16_LE", "-r", "24000", "-c", "1"},
		// ffplay decodes raw s16le as mono by default. Do not pass -ac:
		// FFmpeg 9 removed it (in favor of -ch_layout) and fails to start.
		{"ffplay", "-f", "s16le", "-ar", "24000", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"},
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

// abandon stops a player that failed while receiving audio.
func (p *audioPlayer) abandon() {
	p.stdin.Close()
	if p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
	}
	_ = p.cmd.Wait()
}

// canPlayBuffered reports whether the OS has a player that accepts a complete
// WAV file. macOS ships afplay; Windows ships PowerShell and SoundPlayer.
func canPlayBuffered() bool {
	switch runtime.GOOS {
	case "darwin":
		_, err := exec.LookPath("afplay")
		return err == nil
	case "windows":
		return hasCommand("powershell") || hasCommand("pwsh")
	}
	return false
}

func hasCommand(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

// playBuffered writes a complete WAV and plays it with the system player.
func playBuffered(audio []float32) error {
	return playTempWAV(func(f *os.File) error {
		return wav.Write(f, audio, tts.SampleRate)
	})
}

// playBufferedPCM is playBuffered for already-encoded 16-bit PCM.
func playBufferedPCM(pcm []byte) error {
	return playTempWAV(func(f *os.File) error {
		if _, err := f.Write(wav.Header(tts.SampleRate, len(pcm))); err != nil {
			return err
		}
		_, err := f.Write(pcm)
		return err
	})
}

func playTempWAV(write func(*os.File) error) error {
	f, err := os.CreateTemp("", "pdg-*.wav")
	if err != nil {
		return err
	}
	path := f.Name()
	defer os.Remove(path)
	if err := write(f); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	switch runtime.GOOS {
	case "darwin":
		cmd := exec.Command("afplay", path)
		cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
		return cmd.Run()
	case "windows":
		shell, err := exec.LookPath("powershell")
		if err != nil {
			shell, err = exec.LookPath("pwsh")
			if err != nil {
				return fmt.Errorf("PowerShell not found")
			}
		}
		q := strings.ReplaceAll(path, "'", "''")
		// PowerShell 7 needs the assembly loaded; Windows PowerShell 5.1 has it.
		script := fmt.Sprintf("Add-Type -AssemblyName System.Windows.Extensions -ErrorAction SilentlyContinue; $p = New-Object System.Media.SoundPlayer '%s'; $p.PlaySync()", q)
		cmd := exec.Command(shell, "-NoProfile", "-NonInteractive", "-Command", script)
		cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
		return cmd.Run()
	}
	return fmt.Errorf("buffered playback is not supported on %s", runtime.GOOS)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "pdg:", err)
	os.Exit(1)
}
