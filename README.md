# pdn

`pdn` is a **zero-install, pure-CPU text-to-speech binary**. One static Go
executable turns text into 24 kHz audio: no Python library, no JavaScript
runtime, no ONNX Runtime, no model server, no GPU.

* 24 kHz mono WAV, ~3 M parameter model
* the model, lexicons and vocabulary are embedded with `go:embed`; the binary
  has no runtime data files (`internal/assets/assets_test.go` guards this)
* eSpeak NG for out-of-vocabulary words, either embedded as WASM or called as
  a command
* streaming output and playback that starts in tens of milliseconds
* ~23× realtime scalar, ~30× with Go 1.27 SIMD on a desktop CPU

The model is **Paradee v2g**, the smallest/fastest release.

## Build

Go 1.27 or newer.

```sh
make build          # GOEXPERIMENT=simd (recommended)
make build-scalar   # portable scalar fallback
make build-espeak   # SIMD + embedded eSpeak NG WASM
```

or directly:

```sh
GOEXPERIMENT=simd go build -trimpath -o pdn ./cmd/pdn
go build -trimpath -o pdn ./cmd/pdn          # scalar fallback
```

`-tags espeakwasm` embeds eSpeak NG and runs it in-process with
[wazero](https://wazero.io); see [Licensing](#licensing). `CGO_ENABLED=0`
works.

## Use

```sh
./pdn "Hello, world."                    # writes pdn.wav
./pdn -o hello.wav "Hello, world."
./pdn -speed 1.25 "A little faster."
./pdn -play "Say it out loud."           # pw-play, paplay, aplay or ffplay
echo "Long text..." | ./pdn -o - > out.wav
./pdn -first 4 -chunk 16 -play "Lower latency."
./pdn -seed 42 -o a.wav "Deterministic noise."
```

Flags: `-o` output (`-` = stdout), `-speed` rate, `-seed` noise seed (0 =
random), `-play`, `-quiet`, `-version`, `-first`/`-chunk` (frames in the first
and later streaming chunks; 80 fps, 300 samples each).

## Use as a library

The module root is the public API:

```go
import (
	"context"
	"os"

	"github.com/akhenak/pdgo"
)

e, err := pdgo.New()
if err != nil {
	panic(err)
}
audio, err := e.Synthesize(context.Background(), "Hello from Go.",
	pdgo.WithSpeed(1.1))
if err != nil {
	panic(err)
}
f, _ := os.Create("hello.wav")
defer f.Close()
if err := pdgo.WriteWAV(f, audio); err != nil {
	panic(err)
}
```

`Synthesize` returns mono `float32` at `pdgo.SampleRate` (24 kHz). For early
playback, stream instead:

```go
err = e.Stream(ctx, text, func(chunk []float32) error {
	_, err := player.Write(pdgo.PCM(chunk)) // raw s16le
	return err
}, pdgo.WithChunking(8, 24), pdgo.WithSeed(1))
```

Options: `WithSpeed`, `WithSeed` (reproducible noise), `WithChunking`
(first/later chunk sizes in 80 fps frames). `pdgo.PCM` converts a chunk to
16-bit PCM for a sound device. An `Engine` is safe for concurrent use;
cancellation is checked between chunks. See `examples/hello` for a complete
program.

## Goals

* A **single binary** that does TTS on a fresh Linux machine: no Python
  packages, no Node/Python runtime, no shared inference library to install.
* **Pure CPU inference**, even when that is slower than a GPU path, so there
  are no CUDA/driver/runtime dependencies.
* Keep the default build free of GPL code (see [Licensing](#licensing)).

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
  * default build: the installed `espeak-ng`/`espeak` command as a
    subprocess;
  * `-tags espeakwasm`: a WASI build of eSpeak NG embedded in the binary and
    executed by wazero, so no external command is needed.
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

`pdn -play` pipes 16-bit PCM straight into `pw-play`/`pw-cat`/`paplay`/
`aplay`/`ffplay` as chunks arrive; `-o -` writes a streaming WAV. The decoder
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

The Go code in this repository is **Apache-2.0** (see `LICENSE`). The default
binary does not contain GPL code. It can call an installed `espeak-ng`
command, which runs as a separate process.

**eSpeak NG is GPL-3.0-or-later.** The optional `-tags espeakwasm` build
embeds the eSpeak NG WASI module and its data. That WASM is executed by
wazero as an isolated, interpreted module rather than linked into the Go
binary; on that basis this project keeps its own code Apache-2.0 and treats
the eSpeak artifact as a separate component. This is a project position, not
legal advice: if you redistribute a binary built with `-tags espeakwasm`,
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
