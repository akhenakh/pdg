# PDG

`pdg` is a **zero-install, pure-CPU text-to-speech binary**. One static Go
executable turns text into 24 kHz audio: no Python library, no JavaScript
runtime, no ONNX Runtime, no model server, no GPU.

* 24 kHz mono WAV, ~3 M parameter model
* the model, lexicons, vocabulary, and eSpeak NG fallback are embedded; the
  binary has no runtime data files (`internal/assets/assets_test.go` guards
  the core model/lexicons)
* streaming output; playback starts in tens of milliseconds with a streaming player
* Go 1.27 SIMD inference, scalar fallback available
* ~30× realtime on a desktop CPU

The model is **Paradee v2g**, the smallest/fastest release.

## Install

Prebuilt archives for **Linux, macOS, and Windows** (amd64 and arm64) are on
the [releases page](https://github.com/akhenak/pdg/releases). They are built
with Go 1.27 SIMD and embed eSpeak NG WASM, so there is nothing else to
install.

Homebrew (macOS; Linux users use the release tarball):

```sh
brew install akhenakh/tap/pdg
```

## Build

Go 1.27 or newer.

```sh
make build              # shipped build: SIMD + embedded eSpeak NG WASM
make build-noespeak     # SIMD, no GPL eSpeak; uses espeak-ng from PATH
make build-scalar       # portable fallback, no SIMD, no eSpeak WASM
```

or directly:

```sh
GOEXPERIMENT=simd go build -trimpath -tags espeakwasm -o pdg ./cmd/pdg
GOEXPERIMENT=simd go build -trimpath -o pdg ./cmd/pdg   # without eSpeak WASM
go build -trimpath -o pdg ./cmd/pdg                     # scalar
```

`-tags espeakwasm` embeds eSpeak NG and runs it in-process with
[wazero](https://wazero.io); see [Licensing](#licensing). `CGO_ENABLED=0`
works.

## Use

```sh
./pdg "Hello, world."                    # plays on the default device
./pdg -o hello.wav "Hello, world."       # writes a file instead
./pdg -speed 1.25 "A little faster."
./pdg -first 4 -chunk 16 "Lower latency."
echo "Long text..." | ./pdg -o - > out.wav
./pdg -play=false "Hello."               # writes pdg.wav
./pdg -seed 42 -o a.wav "Deterministic noise."
```

On a headless machine with no audio device, pass `-o FILE` or `-o -`; otherwise
the default is to play.

### Notify when a long command finishes

`pdg` is handy at the end of a long build, test run, or download:

```sh
sleep 20 && pdg "command x finished"
make build && pdg "build finished"
```

Flags: `-o` output file (`-` = stdout; setting it disables playback), `-play`
(default true; set `-play=false` to write `pdg.wav`), `-speed` rate, `-seed`
noise seed (0 = random), `-quiet`, `-version`, `-first`/`-chunk` (frames in the
first and later streaming chunks; 80 fps, 300 samples each).

`-play` streams into `pw-play`, `pw-cat`, `paplay`, `aplay`, or `ffplay` when
one is available. On macOS it falls back to the built-in `afplay`; on Windows
it falls back to PowerShell's `System.Media.SoundPlayer`. Those two need a
complete WAV, so sound starts after synthesis. Install `ffplay` (for example
`brew install ffmpeg`) if you want early playback on macOS or Windows.

## Use as a library

The module root is the public API:

```go
import (
	"context"
	"os"

	"github.com/akhenak/pdg"
)

e, err := pdg.New()
if err != nil {
	panic(err)
}
audio, err := e.Synthesize(context.Background(), "Hello from Go.",
	pdg.WithSpeed(1.1))
if err != nil {
	panic(err)
}
f, _ := os.Create("hello.wav")
defer f.Close()
if err := pdg.WriteWAV(f, audio); err != nil {
	panic(err)
}
```

`Synthesize` returns mono `float32` at `pdg.SampleRate` (24 kHz). For early
playback, stream instead:

```go
err = e.Stream(ctx, text, func(chunk []float32) error {
	_, err := player.Write(pdg.PCM(chunk)) // raw s16le
	return err
}, pdg.WithChunking(8, 24), pdg.WithSeed(1))
```

Options: `WithSpeed`, `WithSeed` (reproducible noise), `WithChunking`
(first/later chunk sizes in 80 fps frames). `pdg.PCM` converts a chunk to
16-bit PCM for a sound device. An `Engine` is safe for concurrent use;
cancellation is checked between chunks. See `examples/hello` for a complete
program.

## Goals

* A **single binary** that does TTS on a fresh machine: no Python packages, no
  Node/Python runtime, no shared inference library to install.
* **Pure CPU inference**, even when that is slower than a GPU path, so there
  are no CUDA/driver/runtime dependencies.
* An **Apache-2.0** build is available; the shipped release additionally
  embeds eSpeak NG as an isolated WASM artifact for names and rare words.

## How it works

```
text ──► misaki-lite G2P ──► phoneme ids ──► Paradee v2g graph ──► 24 kHz WAV
         (Go port)                             (pure-Go executor)
```

* **`tools/convert.py`** folds the released ONNX graph once with ONNX
  Runtime's BASIC optimizer (this expands the int4/int6/int8 weight grids to
  fp32), then writes a compact `PDNG` container: JSON graph + raw tensors.
  The result has 775 dynamic-shape nodes and 12.6 MB of fp32 weights.
* **`internal/engine`** is a small executor for the ~50 ONNX ops used:
  MatMul/Gemm, grouped and depthwise 1-D Conv/ConvTranspose, LayerNorm,
  InstanceNorm, Softmax, Resize, Pad, CumSum, Gather, Slice, the shape
  algebra, and the DSP head. No cgo.
* **`internal/phonemize`** is a Go port of `misaki-lite` from the Paradee web
  bundle: gold/silver lexicons, stress rules, stemming, numbers, currency,
  tokenizer and subtoken groups. For words outside the lexicons it uses
  eSpeak NG:
  * shipped build (`-tags espeakwasm`): a WASI build of eSpeak NG embedded in
    the binary and executed by wazero, no external command;
  * `build-noespeak` / default build: the installed `espeak-ng`/`espeak`
    command as a subprocess, falling back to letter names if absent.
* **`internal/tts`** handles sentence/clause splitting, chunked synthesis and
  concatenation.

`tools/make_testdata.py` and `internal/engine/engine_test.go` compare the Go
executor against ONNX Runtime sample-for-sample. On the three reference
sentences the waveform differs by 0.007–0.05 % RMS, i.e. float rounding.

## Performance

Measured on one AMD Ryzen 9 7900X, CPU only:

| | scalar | Go 1.27 SIMD |
|---|---|---|
| throughput, 4.3 s sentence | ~23× realtime | ~30× realtime |
| first audio, short first sentence | ~60–170 ms | ~50–130 ms |
| first audio, one 30 s sentence | ~1.1 s | ~0.9 s |
| model + lexicons load | ~15 ms | ~15 ms |
| binary | ~21 MB | ~21 MB (28 MB with `espeakwasm`) |

The first-sound number is dominated by the sentence-global prefix (text side,
durations, ConvNeXt trunk), not by the decoder head. Long single sentences
therefore take longer to start than paragraphs, where the first sentence is
usually short; the decoder head itself streams in 150-sample subframes.

## Streaming

The graph is split where the sentence-global work ends:

```
text side + durations + ConvNeXt trunk + noise   (once per sentence)
        │  uT, uY, v2, wH, wL, wO, wZ, y5
        ▼
per-subframe decoder head                         (chunks of 150 samples)
```

`engine.Prepare` runs the prefix, then `StreamState.Chunk(a,b)` runs the head
on a time slice. Streamed audio is **bit-identical** to one-shot synthesis
(`TestStreamingMatchesFull`, `TestStreamMatchesSynthesize`); each chunk carries
one head subframe of overlap. The first chunk is 8 frames (0.1 s) and later
chunks 24 frames (0.3 s) by default.

`pdg` (or `pdg -play`) pipes 16-bit PCM straight into `pw-play`/`pw-cat`/`paplay`/
`aplay`/`ffplay` as chunks arrive (macOS `afplay` and Windows PowerShell are
buffered fallbacks); `-o -` writes a streaming WAV. The decoder
head is already independent of sentence length; the next latency step is
chunking the trunk as well, plus more SIMD in the elementwise ops.

## SIMD

The hot inference kernels have two implementations selected at build time:

* `simd` build (`GOEXPERIMENT=simd`, Go 1.27): vectorized `matmul2D`,
  depthwise convolution, and group-1 convolution using `simd.Float32s`. On
  amd64 this uses AVX/AVX2/AVX-512 when available, NEON on arm64, and the
  package's portable emulation elsewhere.
* scalar build (no experiment): the same loops in portable Go.

`simd` kernels are 3–6× faster in isolation; end-to-end the 30 s sentence
improves by ~1.2×, with the rest of the time going to elementwise ops and
memory traffic. `GODEBUG=simd=0` forces emulation at runtime.

## Why not ncnn

ncnn cannot ingest this graph as shipped:

* `pnnx` (ncnn's current converter) aborts on both the int8 and fp32 ONNX
  files (`map::at` in `pass_level3`).
* `onnx2ncnn` no longer exists in current ncnn.
* the graph relies on ONNX-only quantization nodes and a DSP head with
  `CumSum`, `Resize`, `RandomUniformLike`, and a dynamic-shape sine bank.

A hand-written translator plus custom layers was possible but more fragile
than a focused executor, and cgo would cost the static single-binary install.
The model is ~3 M parameters, so the pure-Go kernels are enough.

## Licensing

The Go code in this repository is **Apache-2.0** (see `LICENSE`).

**eSpeak NG is GPL-3.0-or-later.** The shipped release binaries are built with
`-tags espeakwasm` and embed the eSpeak NG WASI module and its data. That WASM
is executed by wazero as an isolated, interpreted module rather than linked
into the Go binary; on that basis the project keeps its own code Apache-2.0
and treats the eSpeak artifact as a separate component. The default
`build-noespeak`/`build-scalar` binaries contain no eSpeak code. This is a
project position, not legal advice: if you redistribute the embedded build,
satisfy yourself about GPL-3.0 obligations first.

## Credits

Paradee v2g is by **[mesmer.tools](https://mesmer.tools/free-tools/tts-generator)**.
In the author's words:

> I started from Paradee v1.1, a small open model distilled from Kokoro-82M,
> and made it 12.9× faster in the browser while it scored slightly better on
> quality.

Paradee-8M v1.1 is by **Sahil Mahendrakar** (Apache-2.0) and was distilled from
**Kokoro-82M** by **hexgrad** (Apache-2.0, `af_heart` voice). The phonemizer
and lexicons come from [misaki](https://github.com/hexgrad/misaki)
(Apache-2.0). The original pdn engine and release files are mesmer.tools'.
See [LICENSES.md](LICENSES.md).

## Releasing

Push a tag (`vX.Y.Z`). `.github/workflows/release.yml` runs GoReleaser for
Linux, macOS, and Windows on amd64/arm64 with Go 1.27 SIMD and embedded eSpeak
NG, then `scripts/brew-bump.sh` updates `akhenakh/homebrew-tap`. Configure a
`TAP_GITHUB_TOKEN` repository secret with `contents: write` on the tap.
