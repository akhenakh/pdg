GO ?= go
TAGS ?=
BIN ?= pdn

.PHONY: build build-scalar build-espeak build-espeak-scalar test test-scalar test-espeak convert fmt clean

# Go 1.27 SIMD is an experiment: enable it explicitly for the fast build.
build:
	GOEXPERIMENT=simd $(GO) build -trimpath -tags '$(TAGS)' -o $(BIN) ./cmd/pdn

build-scalar:
	$(GO) build -trimpath -tags '$(TAGS)' -o $(BIN) ./cmd/pdn

build-espeak:
	GOEXPERIMENT=simd $(GO) build -trimpath -tags espeakwasm -o $(BIN) ./cmd/pdn

build-espeak-scalar:
	$(GO) build -trimpath -tags espeakwasm -o $(BIN) ./cmd/pdn

test:
	GOEXPERIMENT=simd $(GO) test ./...

test-scalar:
	$(GO) test ./...

test-espeak:
	GOEXPERIMENT=simd $(GO) test -tags espeakwasm ./...

# Regenerate internal/assets/paradee.bin from the released ONNX file.
# Needs: pip install onnx onnxruntime numpy
convert:
	python3 tools/convert.py

fmt:
	gofmt -w ./cmd ./internal

clean:
	rm -f $(BIN) pdn.wav
