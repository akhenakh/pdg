# Licenses

## Project code

The Go code in this repository is **Apache-2.0** (see `LICENSE`). The `pdg` code — including the embedded model, misaki lexicons, and the
scalar or SIMD inference kernels — is distributed under those terms. The
shipped release binaries additionally embed the optional eSpeak WASM
component described below.

## Embedded model and front end

* **Paradee v2g** weights (`internal/assets/paradee.bin`) — Apache-2.0, by
  mesmer.tools. Derived from
  [Paradee-8M v1.1](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.1)
  by Sahil Mahendrakar, distilled from
  [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad
  (`af_heart`).
* **misaki lexicons** (`internal/assets/us_gold.fc.txt`,
  `us_silver.fc.txt`) — Apache-2.0, from
  [hexgrad/misaki](https://github.com/hexgrad/misaki).
* The G2P port is based on `misaki-lite` from the Paradee web bundle
  (Apache-2.0), itself a port of misaki's English G2P.

## eSpeak NG

**eSpeak NG is GPL-3.0-or-later**
(https://github.com/espeak-ng/espeak-ng). It is used only for words outside
the misaki lexicons.

* **Shipped release binaries:** built with `-tags espeakwasm`.
  `internal/phonemize/wasm/espeak.wasm` and `internal/phonemize/wasm/data/`
  contain eSpeak NG and its English data. The module is executed by wazero as
  an isolated, interpreted WASM artifact; it is not linked into the Go binary
  the way a C library would be.
* **`build-noespeak` / `build-scalar`:** no eSpeak code or data is contained
  in the binary; the `espeak-ng` (or `espeak`) executable is invoked as a
  separate process when present.

On that basis this project keeps its own code under Apache-2.0 and treats the
eSpeak WASM artifact as a separate component. This is a project position, not
legal advice. If you redistribute the embedded build, review GPL-3.0
obligations with your own counsel. The `build-noespeak` and `build-scalar`
binaries contain no eSpeak code.

The WASM module is produced by `tools/build_espeak_wasm.sh` from the pinned
eSpeak NG commit plus `tools/espeak_wasm.c`.

## Other dependencies

* [wazero](https://github.com/tetratelabs/wazero) — Apache-2.0 (only linked
  with `-tags espeakwasm`).
* [golang.org/x/text](https://pkg.go.dev/golang.org/x/text) — BSD-3-Clause.
* Go 1.27's `simd` package is part of the Go standard library (BSD-3-Clause)
  and is used only when building with `GOEXPERIMENT=simd`.
