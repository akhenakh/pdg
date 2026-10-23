package phonemize

import (
	"os/exec"
	"strings"
	"sync"
	"unicode"
)

// eSpeak NG is an optional runtime fallback for words that are in neither
// lexicon (mostly names). It is deliberately not bundled: pdn stays a pure Go
// binary, and users who want the upstream-quality fallback can install
// espeak-ng. Without it, unknown words are spelled with letter names.
//
// The conversion below is misaki's own eSpeak-to-misaki table, adapted (like
// the upstream JavaScript) to eSpeak output without tie marks.
type espeakFallback struct {
	path  string
	cache sync.Map // word -> phonemes
}

func newEspeakFallback() *espeakFallback {
	for _, name := range []string{"espeak-ng", "espeak"} {
		if p, err := exec.LookPath(name); err == nil {
			return &espeakFallback{path: p}
		}
	}
	return nil
}

func (e *espeakFallback) available() bool { return e != nil && e.path != "" }

func (e *espeakFallback) phonemize(word string) (string, bool) {
	if !e.available() {
		return "", false
	}
	if v, ok := e.cache.Load(word); ok {
		s := v.(string)
		return s, s != ""
	}
	out, err := exec.Command(e.path, "-q", "--ipa", "-v", "en-us", "--", word).Output()
	s := ""
	if err == nil {
		s = trimSpace(string(out))
		if s != "" {
			s = toMisaki(s)
			// eSpeak can return several candidates; keep the first.
			if i := strings.IndexAny(s, ",\n"); i >= 0 {
				s = trimSpace(s[:i])
			}
		}
	}
	e.cache.Store(word, s)
	return s, s != ""
}

func trimSpace(s string) string { return strings.TrimSpace(s) }

var e2m = [][2]string{
	{"ʔˌn̩", "ʔn"}, {"ʔn̩", "ʔn"},
	{"aɪ", "I"}, {"aʊ", "W"},
	{"dʒ", "ʤ"},
	{"eɪ", "A"}, {"e", "A"},
	{"tʃ", "ʧ"},
	{"ɔɪ", "Y"},
	{"ʲo", "jo"}, {"ʲə", "jə"}, {"ʲ", ""},
	{"ɚ", "əɹ"},
	{"r", "ɹ"},
	{"x", "k"}, {"ç", "k"},
	{"ɬ", "l"},
	{"̃", ""},
}

// toMisaki rewrites eSpeak's IPA into Paradee's phoneme alphabet.
func toMisaki(ps string) string {
	for _, r := range e2m {
		ps = strings.ReplaceAll(ps, r[0], r[1])
	}
	// syllabic consonants: (\S)̩ -> ᵊ$1
	rs := []rune(ps)
	var b strings.Builder
	for i := 0; i < len(rs); i++ {
		if rs[i] == '\u0329' {
			continue
		}
		if i+1 < len(rs) && rs[i+1] == '\u0329' && !unicode.IsSpace(rs[i]) {
			b.WriteRune('ᵊ')
			b.WriteRune(rs[i])
			i++
			continue
		}
		b.WriteRune(rs[i])
	}
	ps = b.String()
	// misaki writes a syllabic l at the end of a word as ᵊl.
	ps = replaceWordFinal(ps, "əl", "ᵊl")
	// misaki keeps ɐ only for the article "a"; elsewhere it is ə.
	ps = replaceExceptWordFinal(ps, "ɐ", "ə")
	ps = strings.ReplaceAll(ps, "oʊ", "O")
	ps = strings.ReplaceAll(ps, "ɜːɹ", "ɜɹ")
	ps = strings.ReplaceAll(ps, "ɜː", "ɜɹ")
	ps = strings.ReplaceAll(ps, "ɪə", "iə")
	ps = strings.ReplaceAll(ps, "ː", "")
	ps = strings.ReplaceAll(ps, "o", "ɔ")
	ps = strings.ReplaceAll(ps, "ɾ", "T")
	ps = strings.ReplaceAll(ps, "ʔ", "t")
	return strings.TrimSpace(ps)
}

func isWordEndAfter(rs []rune, i int) bool {
	if i >= len(rs) {
		return true
	}
	switch rs[i] {
	case ' ', ';', ':', ',', '.', '!', '?', '—', '…', '"', '“', '”', '(', ')':
		return true
	}
	return false
}

func replaceWordFinal(s, old, new string) string {
	ro, rn := []rune(old), []rune(new)
	rs := []rune(s)
	for i := len(rs) - len(ro); i >= 0; i-- {
		if !isWordEndAfter(rs, i+len(ro)) {
			continue
		}
		match := true
		for j := range ro {
			if rs[i+j] != ro[j] {
				match = false
				break
			}
		}
		if match {
			rs = append(append(append([]rune{}, rs[:i]...), rn...), rs[i+len(ro):]...)
		}
	}
	return string(rs)
}

func replaceExceptWordFinal(s, old, new string) string {
	ro, rn := []rune(old), []rune(new)
	rs := []rune(s)
	for i := 0; i < len(rs); i++ {
		if rs[i] != ro[0] {
			continue
		}
		if isWordEndAfter(rs, i+len(ro)) {
			continue
		}
		rs = append(append(append([]rune{}, rs[:i]...), rn...), rs[i+len(ro):]...)
	}
	return string(rs)
}
