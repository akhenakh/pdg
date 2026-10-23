GO ?= go
TAGS ?=
BIN ?= pdn

.PHONY: build build-espeak test test-espeak convert fmt clean

build:
	$(GO) build -trimpath -tags '$(TAGS)' -o $(BIN) ./cmd/pdn

build-espeak:
	$(GO) build -trimpath -tags espeakwasm -o $(BIN) ./cmd/pdn

test:
	$(GO) test ./...

test-espeak:
	$(GO) test -tags espeakwasm ./...

# Regenerate internal/assets/paradee.bin from the released ONNX file.
# Needs: pip install onnx onnxruntime numpy
convert:
	python3 tools/convert.py

fmt:
	gofmt -w ./cmd ./internal

clean:
	rm -f $(BIN) pdn.wav
