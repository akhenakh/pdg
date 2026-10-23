GO ?= go
BIN ?= pdg

.PHONY: build build-noespeak build-scalar build-espeak-scalar test test-noespeak test-scalar convert fmt clean

# The shipped build: Go 1.27 SIMD + embedded eSpeak NG WASM.
build:
	GOEXPERIMENT=simd $(GO) build -trimpath -tags espeakwasm -o $(BIN) ./cmd/pdg

# SIMD, but without the optional GPL eSpeak WASM (uses espeak-ng on PATH).
build-noespeak:
	GOEXPERIMENT=simd $(GO) build -trimpath -o $(BIN) ./cmd/pdg

# Portable fallback, no SIMD and no eSpeak WASM.
build-scalar:
	$(GO) build -trimpath -o $(BIN) ./cmd/pdg

build-espeak-scalar:
	$(GO) build -trimpath -tags espeakwasm -o $(BIN) ./cmd/pdg

test:
	GOEXPERIMENT=simd $(GO) test -tags espeakwasm ./...

test-noespeak:
	GOEXPERIMENT=simd $(GO) test ./...

test-scalar:
	$(GO) test ./...

# Regenerate internal/assets/paradee.bin from the released ONNX file.
# Needs: pip install onnx onnxruntime numpy
convert:
	python3 tools/convert.py

fmt:
	gofmt -w ./cmd ./internal

clean:
	rm -f $(BIN) pdg.wav
