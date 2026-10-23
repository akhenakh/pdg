// Minimal WASI surface for embedding eSpeak NG's phonemizer in Go.
#include <espeak-ng/speak_lib.h>
#include <stdlib.h>
#include <string.h>

static int ready = 0;

__attribute__((export_name("pdn_init")))
int pdn_init(const char *data_path) {
    if (ready) return 0;
    int r = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_path, 0);
    if (r < 0) return r;
    ready = 1;
    return 0;
}

__attribute__((export_name("pdn_phonemize")))
int pdn_phonemize(const char *text, const char *voice, char *out, int cap) {
    if (!ready) return -1;
    espeak_VOICE v;
    memset(&v, 0, sizeof(v));
    v.languages = voice;
    v.name = voice;
    if (espeak_SetVoiceByProperties(&v) != EE_OK) return -2;
    const void *p = text;
    int n = 0;
    out[0] = 0;
    while (p) {
        int term = 0;
        const char *ph = espeak_TextToPhonemesWithTerminator(
            &p, espeakCHARS_AUTO, espeakPHONEMES_IPA, &term);
        if (!ph) break;
        int len = (int)strlen(ph);
        if (n + len + 1 >= cap) return -3;
        memcpy(out + n, ph, len);
        n += len;
        out[n] = 0;
    }
    return n;
}

__attribute__((export_name("pdn_alloc")))
void *pdn_alloc(int n) { return malloc((size_t)n); }

__attribute__((export_name("pdn_free")))
void pdn_free(void *p) { free(p); }

// WASI has no tmpnam; espeak's soundicon uses it but the phonemizer path
// never does.
char *tmpnam(char *s) { (void)s; return NULL; }

// The bundled speech player is C++ and never used on the phonemizer path.
void *__cxa_allocate_exception(unsigned long n) { return malloc((size_t)n); }
void __cxa_throw(void *a, void *b, void *c) { (void)a; (void)b; (void)c; abort(); }
