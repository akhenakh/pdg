package phonemize

import (
	"strings"
	"testing"

	"github.com/akhenak/pdg/internal/assets"
)

// Front-coded lexicons are LF-only in the repository, but a Windows checkout
// can embed them with CRLF; the stray '\r' must not reach phoneme strings.
func TestParseLexiconCRLF(t *testing.T) {
	packed := "0cat\tkˈæt\r\n0dog\tdˈɔɡ\r\n"
	lex := parseLexicon(packed)
	for word, want := range map[string]string{"cat": "kˈæt", "dog": "dˈɔɡ"} {
		v := lex.get(word)
		if v == nil {
			t.Fatalf("%s: missing", word)
		}
		if v.text != want {
			t.Errorf("%s: got %q want %q", word, v.text, want)
		}
	}
}

func TestParseLexiconGoldCRLF(t *testing.T) {
	lf := parseLexicon(assets.Gold)
	crlf := parseLexicon(strings.ReplaceAll(assets.Gold, "\n", "\r\n"))
	if len(lf.keys) != len(crlf.keys) {
		t.Fatalf("keys: lf %d, crlf %d", len(lf.keys), len(crlf.keys))
	}
	for i := range lf.keys {
		if lf.keys[i] != crlf.keys[i] {
			t.Fatalf("key %d: lf %q, crlf %q", i, lf.keys[i], crlf.keys[i])
		}
		if lf.vals[i].text != crlf.vals[i].text {
			t.Fatalf("%s: lf %q, crlf %q", lf.keys[i], lf.vals[i].text, crlf.vals[i].text)
		}
	}
}
