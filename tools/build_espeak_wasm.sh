#!/usr/bin/env bash
# Build the optional embedded eSpeak NG fallback (internal/phonemize/wasm).
#
# Output: internal/phonemize/wasm/espeak.wasm
# The English data files in internal/phonemize/wasm/data are committed; they
# are eSpeak NG's phontab/phonindex/phondata/intonations/en_dict plus
# lang/gmw/en-US (the en-us voice definition).
#
# Requires ~200 MB of downloads and a few minutes. eSpeak NG is GPL-3.0, so
# the resulting module (and any binary built with -tags espeakwasm) is GPL.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/pdg-espeak-build}
WASI_SDK=${WASI_SDK:-$WORK/wasi-sdk}
WASI_VERSION=34.0
ESPEAK_REPO=https://github.com/espeak-ng/espeak-ng.git
ESPEAK_COMMIT=ba90c8e

mkdir -p "$WORK"

if [ ! -x "$WASI_SDK/bin/clang" ]; then
  echo "==> downloading wasi-sdk $WASI_VERSION"
  curl -fL -o "$WORK/wasi-sdk.tar.gz" \
    "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_VERSION%%.*}/wasi-sdk-${WASI_VERSION}-x86_64-linux.tar.gz"
  tar xzf "$WORK/wasi-sdk.tar.gz" -C "$WORK"
  mv "$WORK/wasi-sdk-${WASI_VERSION}-x86_64-linux" "$WASI_SDK"
fi

if [ ! -d "$WORK/espeak-ng" ]; then
  echo "==> cloning eSpeak NG $ESPEAK_COMMIT"
  git clone "$ESPEAK_REPO" "$WORK/espeak-ng"
  git -C "$WORK/espeak-ng" checkout "$ESPEAK_COMMIT"
fi

# WASI's wchar.h declares the isw* functions that eSpeak's compat layer
# aliases to ucd_*; hide the declarations (same workaround the upstream
# Emscripten build uses).
python3 - "$WASI_SDK" <<'PY'
import re, sys, os
p = os.path.join(sys.argv[1], "share/wasi-sysroot/include/wasm32-wasip1/wchar.h")
s = open(p).read()
s = re.sub(r'^int\s+(iswalnum|iswalpha|iswblank|iswcntrl|iswgraph|iswlower|iswprint|iswpunct|iswspace|iswupper|iswxdigit)\(wint_t\);', r'// \g<0>', s, flags=re.M)
open(p, "w").write(s)
PY

echo "==> building libespeak-ng for wasm32-wasip1"
cd "$WORK/espeak-ng"
cmake -B build-wasi \
  -DCMAKE_TOOLCHAIN_FILE="$WASI_SDK/share/cmake/wasi-sdk-p1.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=OFF \
  -DCOMPILE_INTONATIONS=OFF \
  -DBUILD_SHARED_LIBS=OFF >/dev/null
cmake --build build-wasi --target espeak-ng ucd -- -j"$(nproc)" >/dev/null

echo "==> linking wrapper"
"$WASI_SDK/bin/clang++" --target=wasm32-wasip1 \
  --sysroot="$WASI_SDK/share/wasi-sysroot" -O2 \
  -I src/include -I src/ucd-tools/src/include \
  -x c "$ROOT/tools/espeak_wasm.c" -x none \
  build-wasi/src/libespeak-ng/libespeak-ng.a \
  build-wasi/src/ucd-tools/libucd.a \
  build-wasi/src/speechPlayer/libspeechPlayer.a \
  -lm -Wl,--no-entry \
  -Wl,--export=pdn_init -Wl,--export=pdn_phonemize \
  -Wl,--export=pdn_alloc -Wl,--export=pdn_free \
  -o "$ROOT/internal/phonemize/wasm/espeak.wasm"

echo "wrote $ROOT/internal/phonemize/wasm/espeak.wasm"
