# pdn

`pdn` is a text-to-speech binary for Linux (and any Go target) that speaks with
**Paradee v2g** entirely on the CPU. It is one static Go binary: no cgo, no
Python, no ONNX Runtime, no model server.

* 24 kHz mono WAV output
* 2.3 MB ONNX source, expanded to 12.6 MB of fp32 weights in the binary,
  plus ~3.8 MB of misaki lexicons — all embedded
* about 7–9× realtime on one core of a desktop CPU
* `CGO_ENABLED=0` works; cross-compiles like any Go program

The model is the smallest and fastest Paradee release (v2g, fp32 compute),
distilled from Kokoro-82M's `af_heart` voice.

## Build

Go 1.27 or newer:

```sh
go build -trimpath -o pdn ./cmd/pdn
```

Optional: embed eSpeak NG for out-of-vocabulary words (names, rare words).
This removes the runtime dependency on the `espeak-ng` command:

```sh
go build -tags espeakwasm -o pdn ./cmd/pdn
```

`-tags espeakwasm` embeds a WASI build of eSpeak NG and runs it in-process with
[wazero](https://wazero.io). eSpeak NG is **GPL-3.0-or-later**, so a binary
built with that tag is a GPL-3.0 combined work. The default build contains no
GPL code. See [LICENSES.md](LICENSES.md).

## Use

```sh
./pdn "Hello, world."                    # writes pdn.wav
./pdn -o hello.wav "Hello, world."
./pdn -speed 1.25 "A little faster."
./pdn -play "Say it out loud."           # pw-play, paplay, aplay or ffplay
echo "Long text..." | ./pdn -o - > out.wav
./pdn -seed 42 -o a.wav "Deterministic noise."
```

Flags: `-o` output (`-` = stdout), `-speed` rate, `-seed` noise seed (0 =
random), `-play`, `-quiet`, `-version`.

Text is split into sentences the way the model was trained (titles and
initials are kept together), phonemized one sentence at a time, and cut at
clause punctuation when a sentence exceeds the model's 512-token window.

## How it works

```
text ──► misaki-lite G2P ──► phoneme ids ──► Paradee v2g graph ──► 24 kHz WAV
         (Go port)                             (pure-Go executor)
```

* **`tools/convert.py`** runs ONNX Runtime's BASIC optimizer once to
  constant-fold the graph (this expands the int4/int6/int8 weight grids into
  fp32 and removes the tokenizer tables), then writes a `PDNG` container:
  a JSON graph plus a raw tensor blob. The result is 775 nodes with dynamic
  token length, 12.6 MB of fp32 weights.
* **`internal/engine`** is a small dynamic-shape executor for the ~50 ONNX ops
  the graph uses: MatMul/Gemm, 1-D Conv/ConvTranspose (grouped and depthwise),
  LayerNorm, InstanceNorm, Softmax, Resize, Pad, CumSum, Gather, Slice, the
  shape algebra, and the DSP head's Sin/Cos/Erf/Log/noise path. No cgo.
* **`internal/phonemize`** is a faithful Go port of `misaki-lite` from the
  Paradee web bundle: the gold/silver lexicons, stress rules, stemming,
  numbers, currency, the spaCy-like tokenizer, and subtoken groups. Unknown
  words use `espeak-ng` as a subprocess when it is installed (the same mapping
  to misaki's alphabet as the upstream bundle), otherwise letter names.
* **`internal/tts`** handles sentence/clause splitting and concatenation.

`tools/make_testdata.py` and `internal/engine/engine_test.go` compare the Go
executor against ONNX Runtime sample-for-sample. On the three reference
sentences the waveform differs by 0.014–0.039 % RMS (max sample error < 1e-3),
which is float rounding, not a semantic difference.

## Performance

Measured on this machine, one core, after load:

| | |
|---|---|
| synthesis | ~7–9× realtime (v2g, fp32) |
| model + lexicon load | ~15 ms |
| binary | ~21 MB (default), ~28 MB with `-tags espeakwasm` |

The graph is single-threaded today. The obvious next steps are Go 1.27
`simd/archsimd` kernels for the matrix/convolution inner loops and splitting
work across cores; the model is small enough that it is already comfortable
for a CLI.

## Why not ncnn

The short version: ncnn cannot ingest this graph.

* `pnnx` (ncnn's current converter) aborts on both the int8 and fp32 ONNX
  files (`map::at` in `pass_level3`, after failing on double constants).
* `onnx2ncnn` no longer exists in current ncnn.
* The graph relies on ONNX-only pieces: `DynamicQuantizeLinear` /
  `MatMulInteger` in the int8 file, int4 weight unpacking baked into the graph,
  and a DSP head with `CumSum`, `Resize`, `RandomUniformLike` and a
  dynamic-shape sine bank.

A hand-written ONNX→ncnn translator plus custom layers was possible but far
more fragile than a focused executor, and cgo would have cost the static
single-binary install. The model is ~3 M parameters, so the pure-Go kernels are
fast enough without a general-purpose inference runtime.

## Streaming

v2g is architecturally streamable (no LSTM, fixed per-channel norms), and the
upstream `pdn` engine emits audio in 80 fps chunks. This port currently runs
one sentence at a time through the whole graph, so first audio waits for the
sentence. The graph is already split-friendly; a chunked decoder is the next
step if sub-100 ms first sound matters.

## Model and credits

Paradee v2g is Apache-2.0. It is derived from
[Paradee-8M v1.1](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.1) by
Sahil Mahendrakar, itself distilled from
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad
(`af_heart`). The phonemizer and lexicons come from
[misaki](https://github.com/hexgrad/misaki) (Apache-2.0). The original engine,
JavaScript, and release files are by mesmer.tools. See
[LICENSES.md](LICENSES.md).
