//go:build !espeakwasm

package phonemize

// The default build keeps eSpeak NG out of the binary: unknown words use the
// installed espeak-ng command when available, else letter spelling.

func newWASMFallback() wasmFallback { return nil }
