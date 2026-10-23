//go:build espeakwasm

package phonemize

// Embedded eSpeak NG fallback. Build with -tags espeakwasm to include it.
//
// The eSpeak NG library and its data are GPL-3.0-or-later, so a binary built
// with this tag is a GPL-3.0 combined work. The default build (no tag) does
// not include any GPL code; see LICENSES.md.
//
// The wasm module is built by tools/build_espeak_wasm.sh and exposes
// pdn_init / pdn_phonemize / pdn_alloc / pdn_free (see that script).

import (
	wctx "context"
	"embed"
	"fmt"
	"io/fs"
	"sync"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

//go:embed wasm/espeak.wasm
var wasmBytes []byte

//go:embed all:wasm/data
var wasmData embed.FS

type wasmEspeak struct {
	mu    sync.Mutex
	mod   api.Module
	mem   api.Memory
	alloc api.Function
	free  api.Function
	phon  api.Function
	ctx   wctx.Context
	cache sync.Map
}

// lazyWASM defers engine creation (and wazero's compilation) until a word
// actually misses both lexicons.
type lazyWASM struct {
	once sync.Once
	e    *wasmEspeak
}

func newWASMFallback() wasmFallback { return &lazyWASM{} }

func (l *lazyWASM) phonemize(word string) (string, bool) {
	l.once.Do(func() { l.e = newWASMEngine() })
	if l.e == nil {
		return "", false
	}
	return l.e.phonemize(word)
}

func newWASMEngine() *wasmEspeak {
	ctx := wctx.Background()
	r := wazero.NewRuntime(ctx)
	if _, err := wasi_snapshot_preview1.Instantiate(ctx, r); err != nil {
		r.Close(ctx)
		return nil
	}
	data, err := fs.Sub(wasmData, "wasm/data")
	if err != nil {
		r.Close(ctx)
		return nil
	}
	cfg := wazero.NewModuleConfig().
		WithFSConfig(wazero.NewFSConfig().WithFSMount(data, "/")).
		WithStartFunctions()
	mod, err := r.InstantiateWithConfig(ctx, wasmBytes, cfg)
	if err != nil {
		r.Close(ctx)
		return nil
	}
	e := &wasmEspeak{
		ctx:   ctx,
		mod:   mod,
		mem:   mod.Memory(),
		alloc: mod.ExportedFunction("pdn_alloc"),
		free:  mod.ExportedFunction("pdn_free"),
		phon:  mod.ExportedFunction("pdn_phonemize"),
	}
	if e.alloc == nil || e.free == nil || e.phon == nil {
		r.Close(ctx)
		return nil
	}
	p, err := e.writeString("/")
	if err != nil {
		r.Close(ctx)
		return nil
	}
	res, err := mod.ExportedFunction("pdn_init").Call(ctx, uint64(p))
	if err != nil || res[0] != 0 {
		r.Close(ctx)
		return nil
	}
	return e
}

func (e *wasmEspeak) writeString(s string) (uint32, error) {
	res, err := e.alloc.Call(e.ctx, uint64(len(s)+1))
	if err != nil {
		return 0, err
	}
	p := uint32(res[0])
	if !e.mem.Write(p, []byte(s)) {
		return 0, fmt.Errorf("write out of bounds")
	}
	e.mem.WriteByte(p+uint32(len(s)), 0)
	return p, nil
}

func (e *wasmEspeak) phonemize(word string) (string, bool) {
	if v, ok := e.cache.Load(word); ok {
		s := v.(string)
		return s, s != ""
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	textPtr, err := e.writeString(word)
	if err != nil {
		return "", false
	}
	defer e.free.Call(e.ctx, uint64(textPtr))
	voicePtr, err := e.writeString("en-us")
	if err != nil {
		return "", false
	}
	defer e.free.Call(e.ctx, uint64(voicePtr))
	const cap = 8192
	bufRes, err := e.alloc.Call(e.ctx, cap)
	if err != nil {
		return "", false
	}
	buf := uint32(bufRes[0])
	defer e.free.Call(e.ctx, uint64(buf))
	res, err := e.phon.Call(e.ctx, uint64(textPtr), uint64(voicePtr), uint64(buf), cap)
	if err != nil || int32(res[0]) <= 0 {
		e.cache.Store(word, "")
		return "", false
	}
	b, ok := e.mem.Read(buf, uint32(res[0]))
	if !ok {
		return "", false
	}
	s := toMisaki(string(b))
	e.cache.Store(word, s)
	return s, s != ""
}
