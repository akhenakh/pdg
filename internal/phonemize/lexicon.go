package phonemize

import (
	"encoding/json"
	"sort"
	"strings"
)

// lexValue is either a plain phoneme string or one of misaki's tagged
// entries ("DEFAULT", "None", "NOUN" ...).
type lexValue struct {
	text string
	tags map[string]*string
}

func (v *lexValue) pick(ctx *context) *string {
	if v.tags == nil {
		s := v.text
		return &s
	}
	if ctx != nil && ctx.futureVowel == nil {
		if s, ok := v.tags["None"]; ok {
			return s
		}
	}
	if s, ok := v.tags["DEFAULT"]; ok {
		return s
	}
	s := ""
	return &s
}

// lexicon is misaki's dictionary stored as a front-coded, sorted file:
//
//	<p><key suffix>\t<phonemes>
//
// where <p> is the number of bytes shared with the previous key.
type lexicon struct {
	keys []string
	vals []*lexValue
}

func parseLexicon(text string) *lexicon {
	lines := strings.Split(strings.TrimSuffix(text, "\n"), "\n")
	lex := &lexicon{keys: make([]string, 0, len(lines)), vals: make([]*lexValue, 0, len(lines))}
	prev := ""
	for _, line := range lines {
		if line == "" {
			continue
		}
		tab := strings.IndexByte(line, '\t')
		if tab < 0 {
			continue
		}
		n := int(line[0] - '0')
		if n > len(prev) {
			n = len(prev)
		}
		key := prev[:n] + line[1:tab]
		val := line[tab+1:]
		v := &lexValue{text: val}
		if strings.HasPrefix(val, "{") {
			var tags map[string]*string
			if err := json.Unmarshal([]byte(val), &tags); err == nil {
				v.tags = tags
			}
		}
		lex.keys = append(lex.keys, key)
		lex.vals = append(lex.vals, v)
		prev = key
	}
	return lex
}

func (l *lexicon) raw(w string) *lexValue {
	i := sort.SearchStrings(l.keys, w)
	if i < len(l.keys) && l.keys[i] == w {
		return l.vals[i]
	}
	return nil
}

func (l *lexicon) get(w string) *lexValue {
	if v := l.raw(w); v != nil || len([]rune(w)) < 2 {
		return v
	}
	lo := strings.ToLower(w)
	rs := []rune(w)
	cap := strings.ToUpper(string(rs[0])) + strings.ToLower(string(rs[1:]))
	if w == cap && w != lo {
		return l.raw(lo)
	}
	if w == lo && w != cap {
		return l.raw(cap)
	}
	return nil
}

func (l *lexicon) has(w string) bool { return l.get(w) != nil }
