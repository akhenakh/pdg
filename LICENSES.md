# Licenses

`pdn`'s Go code is Apache-2.0 (see `LICENSE`).

## Embedded model and front end

* **Paradee v2g** weights (`internal/assets/paradee.bin`) — Apache-2.0.
  Derived from [Paradee-8M v1.1](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.1)
  by Sahil Mahendrakar, distilled from
  [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad
  (`af_heart`). The v2 release files are by mesmer.tools.
* **misaki lexicons** (`internal/assets/us_gold.fc.txt`,
  `us_silver.fc.txt`) — Apache-2.0, from
  [hexgrad/misaki](https://github.com/hexgrad/misaki).
* The G2P port is based on `misaki-lite` from the Paradee web bundle
  (Apache-2.0), itself a port of misaki's English G2P.

## Optional eSpeak NG fallback (`-tags espeakwasm`)

`internal/phonemize/wasm/espeak.wasm` and `internal/phonemize/wasm/data/`
contain **eSpeak NG**, which is **GPL-3.0-or-later**
(https://github.com/espeak-ng/espeak-ng). Neither is part of the default
build: without `-tags espeakwasm` the binary does not contain eSpeak code or
data and can use an installed `espeak-ng` command as a subprocess instead.

A binary built with `-tags espeakwasm` combines Apache-2.0 and GPL-3.0 code
and must be distributed under the GPL-3.0. Build it only if that is acceptable
for your distribution.

The wasm module is produced by `tools/build_espeak_wasm.sh` from the pinned
eSpeak NG commit plus `tools/espeak_wasm.c`. wazero is Apache-2.0/MIT.

## Other dependencies

* [wazero](https://github.com/tetratelabs/wazero) — Apache-2.0 (only linked
  when building with `-tags espeakwasm`).
* [golang.org/x/text](https://pkg.go.dev/golang.org/x/text) — BSD-3-Clause.
