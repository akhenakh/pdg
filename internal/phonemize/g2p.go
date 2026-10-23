// Package phonemize is a Go port of misaki-lite, the American-English
// grapheme-to-phoneme front end Paradee was trained on. It is a faithful port
// of the JavaScript in mesmer.tools' Paradee bundle, minus the optional eSpeak
// NG fallback: unknown words are spelled with their letter names instead.
package phonemize

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"

	"github.com/akhenak/paradee-go/internal/assets"
)

const (
	primary   = 'ˈ'
	secondary = 'ˌ'
)

var (
	vowels      = runeSet("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")
	consonants  = runeSet("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ")
	usTaus      = runeSet("AIOWYiuæɑəɛɪɹʊʌ")
	puncts      = runeSet(";:,.!?—…\"“”")
	nonQuote    = runeSet(";:,.!?—…")
	diphthongs  = runeSet("AIOQWYʤʧ")
	subtokenJnk = runeSet("',-._‘’/")
	stresses    = []rune{secondary, primary}
)

var (
	symbols = map[string]string{
		"%": "percent", "&": "and", "+": "plus", "@": "at",
	}
	currencyUnits = map[rune][2]string{
		'$': {"dollar", "cent"},
		'£': {"pound", "pence"},
		'€': {"euro", "cent"},
	}
	ordinals = []string{"st", "nd", "rd", "th"}
)

func runeSet(s string) map[rune]bool {
	m := make(map[rune]bool, len(s))
	for _, r := range s {
		m[r] = true
	}
	return m
}

type context struct {
	futureVowel *bool
	nextIsWord  bool
}

// Engine converts English text to misaki phonemes and model token ids.
type wasmFallback interface {
	phonemize(string) (string, bool)
}

type Engine struct {
	gold   *lexicon
	silver *lexicon
	vocab  map[rune]int64
	espeak *espeakFallback
	wasm   wasmFallback
}

// New builds the phonemizer from the embedded lexicons and vocabulary.
func New() (*Engine, error) {
	e := &Engine{
		gold:   parseLexicon(assets.Gold),
		silver: parseLexicon(assets.Silver),
		espeak: newEspeakFallback(),
		wasm:   newWASMFallback(),
	}
	var cfg struct {
		Vocab map[string]int64 `json:"vocab"`
	}
	if err := json.Unmarshal(assets.Config, &cfg); err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}
	e.vocab = make(map[rune]int64, len(cfg.Vocab))
	for k, v := range cfg.Vocab {
		r := []rune(k)
		if len(r) == 1 {
			e.vocab[r[0]] = v
		}
	}
	return e, nil
}

// Phonemize converts one sentence to a phoneme string. Text is not split;
// callers split sentences first, as the model was trained per sentence.
func (e *Engine) Phonemize(text string) string {
	return e.g2pOnce(text)
}

// IDs maps phonemes to model ids: '$' (0) at both ends, characters outside
// the vocabulary dropped, and the whole sequence capped at 512 ids.
func (e *Engine) IDs(phonemes string) []int64 {
	ids := make([]int64, 0, len(phonemes)+2)
	ids = append(ids, 0)
	for _, r := range phonemes {
		if v, ok := e.vocab[r]; ok {
			ids = append(ids, v)
		}
	}
	ids = append(ids, 0)
	if len(ids) > 512 {
		ids = ids[:512]
	}
	return ids
}

// ---- stress -------------------------------------------------------------

func isStress(r rune) bool { return r == primary || r == secondary }

func applyStress(ps string, stress *float64) string {
	if stress == nil {
		return ps
	}
	s := *stress
	if s < -1 {
		return strings.NewReplacer(string(primary), "", string(secondary), "").Replace(ps)
	}
	if s == -1 || ((s == 0 || s == -0.5) && strings.ContainsRune(ps, primary)) {
		ps = strings.ReplaceAll(ps, string(secondary), "")
		return strings.ReplaceAll(ps, string(primary), string(secondary))
	}
	hasS := false
	hasV := false
	for _, r := range ps {
		if isStress(r) {
			hasS = true
		}
		if vowels[r] {
			hasV = true
		}
	}
	if (s == 0 || s == 0.5 || s == 1) && !hasS {
		if hasV {
			return restress(string(secondary) + ps)
		}
		return ps
	}
	if s >= 1 && !strings.ContainsRune(ps, primary) && strings.ContainsRune(ps, secondary) {
		return strings.ReplaceAll(ps, string(secondary), string(primary))
	}
	if s > 1 && !hasS {
		if hasV {
			return restress(string(primary) + ps)
		}
	}
	return ps
}

// restress moves every stress mark in front of the next vowel.
func restress(ps string) string {
	rs := []rune(ps)
	type insert struct {
		at    int
		order int
		ch    rune
	}
	var inserts []insert
	moved := make(map[int]bool)
	for i, r := range rs {
		if !isStress(r) {
			continue
		}
		j := i + 1
		for j < len(rs) && !vowels[rs[j]] {
			j++
		}
		if j < len(rs) {
			inserts = append(inserts, insert{at: j, order: i, ch: r})
			moved[i] = true
		}
	}
	out := make([]rune, 0, len(rs)+len(inserts))
	for i := 0; i <= len(rs); i++ {
		for _, in := range inserts {
			if in.at == i {
				out = append(out, in.ch)
			}
		}
		if i < len(rs) && !moved[i] {
			out = append(out, rs[i])
		}
	}
	return string(out)
}

func stressWeight(ps string) int {
	w := 0
	for _, r := range ps {
		if diphthongs[r] {
			w += 2
		} else {
			w++
		}
	}
	return w
}

func ptrF(v float64) *float64 { return &v }
func ptrS(s string) *string   { return &s }

// ---- lexicon lookup -----------------------------------------------------

func (e *Engine) getNNP(word string) *string {
	var out []rune
	for _, c := range word {
		if !unicode.IsLetter(c) {
			continue
		}
		v := e.gold.get(strings.ToUpper(string(c)))
		// misaki only spells letters whose entry is a plain string, not a
		// tagged one; the rest fall through to eSpeak.
		if v == nil || v.tags != nil {
			return nil
		}
		out = append(out, []rune(v.text)...)
	}
	s := applyStress(string(out), ptrF(0))
	k := strings.LastIndex(s, string(secondary))
	if k < 0 {
		return &s
	}
	return ptrS(s[:k] + string(primary) + s[k+1:])
}

func (e *Engine) lookup(word string, stress *float64, ctx *context) *string {
	if word == strings.ToUpper(word) && !e.gold.has(word) {
		word = strings.ToLower(word)
	}
	v := e.gold.get(word)
	if v == nil {
		v = e.silver.get(word)
	}
	var ps *string
	if v != nil {
		if v.tags != nil {
			if word == strings.ToUpper(word) && len([]rune(word)) > 1 {
				if n, ok := v.tags["NOUN"]; ok && n == nil {
					ps = nil
				} else {
					ps = v.pick(ctx)
				}
			} else {
				ps = v.pick(ctx)
			}
		} else {
			ps = ptrS(v.text)
		}
	}
	if ps == nil {
		if p := e.getNNP(word); p != nil {
			return p
		}
		if e.wasm != nil {
			if s, ok := e.wasm.phonemize(word); ok {
				return &s
			}
		}
		if e.espeak.available() {
			if s, ok := e.espeak.phonemize(word); ok {
				return &s
			}
		}
		return nil
	}
	return ptrS(applyStress(*ps, stress))
}

func (e *Engine) isKnown(word string) bool {
	if e.gold.has(word) || e.silver.has(word) {
		return true
	}
	if _, ok := symbols[word]; ok {
		return true
	}
	if !isAlphaWord(word) {
		return false
	}
	n := len([]rune(word))
	if n == 1 {
		return true
	}
	if word == strings.ToUpper(word) && e.gold.has(strings.ToLower(word)) {
		return true
	}
	rs := []rune(word)
	return string(rs[1:]) == strings.ToUpper(string(rs[1:]))
}

func isAlphaWord(w string) bool {
	if w == "" {
		return false
	}
	for _, r := range w {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return true
}

func isDigitWord(w string) bool {
	if w == "" {
		return false
	}
	for _, r := range w {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func lastRune(s string) rune {
	rs := []rune(s)
	if len(rs) == 0 {
		return 0
	}
	return rs[len(rs)-1]
}

func dropLast(s string, n int) string {
	rs := []rune(s)
	if n > len(rs) {
		n = len(rs)
	}
	return string(rs[:len(rs)-n])
}

func (e *Engine) sSuffix(st string, stress *float64, ctx *context) *string {
	if st == "" {
		return nil
	}
	p := e.lookup(st, stress, ctx)
	if p == nil {
		return nil
	}
	l := lastRune(*p)
	var out string
	switch {
	case strings.ContainsRune("ptkfθ", l):
		out = *p + "s"
	case strings.ContainsRune("szʃʒʧʤ", l):
		out = *p + "ᵻz"
	default:
		out = *p + "z"
	}
	return &out
}

func (e *Engine) stemS(w string, stress *float64, ctx *context) *string {
	if len([]rune(w)) < 3 || !strings.HasSuffix(w, "s") {
		return nil
	}
	var st string
	switch {
	case !strings.HasSuffix(w, "ss") && e.isKnown(dropLast(w, 1)):
		st = dropLast(w, 1)
	case (strings.HasSuffix(w, "'s") || (len([]rune(w)) > 4 && strings.HasSuffix(w, "es") && !strings.HasSuffix(w, "ies"))) && e.isKnown(dropLast(w, 2)):
		st = dropLast(w, 2)
	case len([]rune(w)) > 4 && strings.HasSuffix(w, "ies") && e.isKnown(dropLast(w, 3)+"y"):
		st = dropLast(w, 3) + "y"
	default:
		return nil
	}
	return e.sSuffix(st, stress, ctx)
}

func edSuffix(st *string) *string {
	if st == nil {
		return nil
	}
	s := *st
	l := lastRune(s)
	switch {
	case strings.ContainsRune("pkfθʃsʧ", l):
		out := s + "t"
		return &out
	case l == 'd':
		out := s + "ᵻd"
		return &out
	}
	if l != 't' {
		out := s + "d"
		return &out
	}
	if len([]rune(s)) < 2 {
		out := s + "ɪd"
		return &out
	}
	rs := []rune(s)
	if usTaus[rs[len(rs)-2]] {
		out := string(rs[:len(rs)-1]) + "ɾᵻd"
		return &out
	}
	out := s + "ᵻd"
	return &out
}

func (e *Engine) stemEd(w string, stress *float64, ctx *context) *string {
	if len([]rune(w)) < 4 || !strings.HasSuffix(w, "d") {
		return nil
	}
	var st string
	switch {
	case !strings.HasSuffix(w, "dd") && e.isKnown(dropLast(w, 1)):
		st = dropLast(w, 1)
	case len([]rune(w)) > 4 && strings.HasSuffix(w, "ed") && !strings.HasSuffix(w, "eed") && e.isKnown(dropLast(w, 2)):
		st = dropLast(w, 2)
	default:
		return nil
	}
	return edSuffix(e.lookup(st, stress, ctx))
}

func ingSuffix(st *string) *string {
	if st == nil {
		return nil
	}
	s := *st
	rs := []rune(s)
	if len(rs) > 1 && rs[len(rs)-1] == 't' && usTaus[rs[len(rs)-2]] {
		out := string(rs[:len(rs)-1]) + "ɾɪŋ"
		return &out
	}
	out := s + "ɪŋ"
	return &out
}

func (e *Engine) stemIng(w string, stress *float64, ctx *context) *string {
	if len([]rune(w)) < 5 || !strings.HasSuffix(w, "ing") {
		return nil
	}
	var st string
	switch {
	case len([]rune(w)) > 5 && e.isKnown(dropLast(w, 3)):
		st = dropLast(w, 3)
	case e.isKnown(dropLast(w, 3) + "e"):
		st = dropLast(w, 3) + "e"
	case len([]rune(w)) > 5 && doubleConsonantIng(w) && e.isKnown(dropLast(w, 4)):
		st = dropLast(w, 4)
	default:
		return nil
	}
	return ingSuffix(e.lookup(st, stress, ctx))
}

// doubleConsonantIng matches /([bcdgklmnprstvxz])\1ing$|cking$/ without
// backreferences (Go's regexp has none).
func doubleConsonantIng(w string) bool {
	rs := []rune(w)
	n := len(rs)
	if n >= 4 && string(rs[n-4:]) == "cking" {
		return true
	}
	if n < 5 || string(rs[n-3:]) != "ing" {
		return false
	}
	c := rs[n-4]
	return strings.ContainsRune("bcdgklmnprstvxz", c) && rs[n-5] == c
}

func (e *Engine) specialCase(word string, ctx *context) *string {
	if s, ok := symbols[word]; ok {
		return e.lookup(s, nil, ctx)
	}
	if strings.Contains(strings.Trim(word, "."), ".") && isAlphaWord(strings.ReplaceAll(word, ".", "")) {
		parts := strings.Split(word, ".")
		longest := 0
		for _, p := range parts {
			if n := len([]rune(p)); n > longest {
				longest = n
			}
		}
		if longest < 3 {
			return e.getNNP(word)
		}
	}
	switch word {
	case "a", "A":
		if ctx.nextIsWord {
			return ptrS("ɐ")
		}
		return ptrS("ˈA")
	case "am", "Am", "AM":
		if ctx.futureVowel == nil || word != "am" {
			return e.gold.get("am").pick(nil)
		}
		return ptrS("ɐm")
	case "an", "An", "AN":
		return ptrS("ɐn")
	case "I":
		return ptrS("ˌI")
	case "to", "To":
		if ctx.futureVowel == nil {
			return e.gold.get("to").pick(nil)
		}
		if *ctx.futureVowel {
			return ptrS("tʊ")
		}
		return ptrS("tə")
	case "in", "In", "IN":
		p := "ɪn"
		if ctx.futureVowel == nil {
			p = "ˈɪn"
		}
		return &p
	case "the", "The", "THE":
		if ctx.futureVowel != nil && *ctx.futureVowel {
			return ptrS("ði")
		}
		return ptrS("ðə")
	case "used", "Used", "USED":
		return e.gold.get("used").tags["DEFAULT"]
	}
	if regexVS.MatchString(word) {
		return e.lookup("versus", nil, ctx)
	}
	return nil
}

var regexVS = regexp.MustCompile(`(?i)^vs\.?$`)

func (e *Engine) getWord(word string, stress *float64, ctx *context) *string {
	if sc := e.specialCase(word, ctx); sc != nil {
		return sc
	}
	wl := strings.ToLower(word)
	rs := []rune(word)
	if len(rs) > 1 && isAlphaWord(strings.ReplaceAll(word, "'", "")) && word != wl && !e.gold.has(word) && !e.silver.has(word) {
		allCaps := word == strings.ToUpper(word)
		title := string(rs[1:]) == strings.ToLower(string(rs[1:]))
		if allCaps || title {
			if e.gold.has(wl) || e.silver.has(wl) || e.stemS(wl, stress, ctx) != nil ||
				e.stemEd(wl, stress, ctx) != nil || e.stemIng(wl, stress, ctx) != nil {
				word = wl
			}
		}
	}
	if e.isKnown(word) {
		return e.lookup(word, stress, ctx)
	}
	if strings.HasSuffix(word, "s'") && e.isKnown(dropLast(word, 2)+"'s") {
		return e.lookup(dropLast(word, 2)+"'s", stress, ctx)
	}
	if strings.HasSuffix(word, "'") && e.isKnown(dropLast(word, 1)) {
		return e.lookup(dropLast(word, 1), stress, ctx)
	}
	if p := e.stemS(word, stress, ctx); p != nil {
		return p
	}
	if p := e.stemEd(word, stress, ctx); p != nil {
		return p
	}
	sp := 0.5
	if stress != nil {
		sp = *stress
	}
	return e.stemIng(word, &sp, ctx)
}

// ---- numbers ------------------------------------------------------------

var ones = strings.Split("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen", " ")
var tens = strings.Split(",,twenty,thirty,forty,fifty,sixty,seventy,eighty,ninety", ",")

var scales = []struct {
	v float64
	w string
}{
	{1e12, "trillion"}, {1e9, "billion"}, {1e6, "million"}, {1e3, "thousand"},
}

func cardinal(n int64) []string {
	switch {
	case n < 20:
		return []string{ones[n]}
	case n < 100:
		if n%10 != 0 {
			return []string{tens[n/10], ones[n%10]}
		}
		return []string{tens[n/10]}
	case n < 1000:
		out := []string{ones[n/100], "hundred"}
		if n%100 != 0 {
			out = append(out, cardinal(n%100)...)
		}
		return out
	case n < 1e15:
		for _, sc := range scales {
			v := int64(sc.v)
			if n >= v {
				h := n / v
				r := n - h*v
				out := append(cardinal(h), sc.w)
				if r != 0 {
					out = append(out, cardinal(r)...)
				}
				return out
			}
		}
	}
	var out []string
	for _, d := range strconv.FormatInt(n, 10) {
		out = append(out, ones[d-'0'])
	}
	return out
}

func ordinal(n int64) []string {
	w := cardinal(n)
	l := w[len(w)-1]
	w = w[:len(w)-1]
	irr := map[string]string{"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
	if v, ok := irr[l]; ok {
		return append(w, v)
	}
	if strings.HasSuffix(l, "y") {
		return append(w, l[:len(l)-1]+"ieth")
	}
	return append(w, l+"th")
}

func year(n int64) []string {
	hi, lo := n/100, n%100
	if hi == 0 || (hi%10 == 0 && lo < 10) || hi >= 100 {
		return cardinal(n)
	}
	out := cardinal(hi)
	if lo == 0 {
		return append(out, "hundred")
	}
	if lo < 10 {
		return append(append(out, "oh"), cardinal(lo)...)
	}
	return append(out, cardinal(lo)...)
}

func isNumber(word string, isHead bool) bool {
	if !strings.ContainsAny(word, "0123456789") {
		return false
	}
	rs := []rune(word)
	suffixes := append([]string{"ing", "'d", "ed", "'s"}, append(ordinals, "s")...)
	for _, s := range suffixes {
		if strings.HasSuffix(word, s) {
			rs = rs[:len(rs)-len([]rune(s))]
			break
		}
	}
	for i, c := range rs {
		if c >= '0' && c <= '9' {
			continue
		}
		if c == ',' || c == '.' {
			continue
		}
		if isHead && i == 0 && c == '-' {
			continue
		}
		return false
	}
	return true
}

var regexSuffixLetters = regexp.MustCompile(`[a-z']+$`)

func (e *Engine) getNumber(word, currency string, isHead bool) *string {
	suffix := ""
	if m := regexSuffixLetters.FindString(word); m != "" {
		suffix = m
		word = word[:len(word)-len(m)]
	}
	var res []*string
	ext := func(ws []string, stress *float64) {
		for _, w := range ws {
			st := stress
			if w == "point" {
				st = ptrF(-2)
			}
			res = append(res, e.lookup(w, st, nil))
		}
	}
	digitsOf := func(s string) {
		for _, d := range s {
			ext(cardinal(int64(d-'0')), nil)
		}
	}
	units, hasCurrency := currencyUnits[firstRune(currency)]
	switch {
	case isDigitWord(word) && containsString(ordinals, suffix):
		n, _ := strconv.ParseInt(word, 10, 64)
		ext(ordinal(n), nil)
	case len(res) == 0 && len([]rune(word)) == 4 && !hasCurrency && isDigitWord(word):
		n, _ := strconv.ParseInt(word, 10, 64)
		ext(year(n), nil)
	case !isHead && !strings.Contains(word, "."):
		num := strings.ReplaceAll(word, ",", "")
		rs := []rune(num)
		switch {
		case rs[0] == '0' || len(rs) > 3:
			digitsOf(num)
		case len(rs) == 3 && !strings.HasSuffix(num, "00"):
			ext(cardinal(int64(rs[0]-'0')), nil)
			if rs[1] == '0' {
				res = append(res, e.lookup("O", ptrF(-2), nil))
				ext(cardinal(int64(rs[2]-'0')), nil)
			} else {
				n, _ := strconv.ParseInt(string(rs[1:]), 10, 64)
				ext(cardinal(n), nil)
			}
		default:
			n, _ := strconv.ParseInt(num, 10, 64)
			ext(cardinal(n), nil)
		}
	case strings.Count(word, ".") > 1 || !isHead:
		for _, num := range strings.Split(strings.ReplaceAll(word, ",", ""), ".") {
			if num == "" {
				continue
			}
			rs := []rune(num)
			if rs[0] == '0' || (len(rs) != 2 && hasNonZero(rs[1:])) {
				digitsOf(num)
			} else {
				n, _ := strconv.ParseInt(num, 10, 64)
				ext(cardinal(n), nil)
			}
		}
	case hasCurrency && (!strings.Contains(word, ".") || len(strings.Split(word, ".")[1]) < 3):
		type pair struct {
			n    int64
			unit string
		}
		var pairs []pair
		for i, nstr := range strings.Split(strings.ReplaceAll(word, ",", ""), ".") {
			if i >= 2 {
				break
			}
			var n int64
			if nstr != "" {
				n, _ = strconv.ParseInt(nstr, 10, 64)
			}
			if units[i] != "" {
				pairs = append(pairs, pair{n, units[i]})
			}
		}
		if len(pairs) > 1 {
			if pairs[1].n == 0 {
				pairs = pairs[:1]
			} else if pairs[0].n == 0 {
				pairs = pairs[1:]
			}
		}
		for i, p := range pairs {
			if i > 0 {
				res = append(res, e.lookup("and", nil, nil))
			}
			ext(cardinal(p.n), nil)
			if abs64(p.n) != 1 && p.unit != "pence" {
				res = append(res, e.stemS(p.unit+"s", nil, nil))
			} else {
				res = append(res, e.lookup(p.unit, nil, nil))
			}
		}
	case !strings.Contains(word, "."):
		n, _ := strconv.ParseInt(strings.ReplaceAll(word, ",", ""), 10, 64)
		if containsString(ordinals, suffix) {
			ext(ordinal(n), nil)
		} else {
			ext(cardinal(n), nil)
		}
	default:
		cleaned := strings.ReplaceAll(word, ",", "")
		f, err := strconv.ParseFloat(cleaned, 64)
		if err != nil {
			return nil
		}
		parts := strings.Split(strconv.FormatFloat(f, 'f', -1, 64), ".")
		integer := parts[0]
		frac := ""
		if len(parts) > 1 {
			frac = parts[1]
		}
		var ws []string
		if !strings.HasPrefix(word, ".") {
			n, _ := strconv.ParseInt(integer, 10, 64)
			ws = append(ws, cardinal(n)...)
		}
		ws = append(ws, "point")
		if strings.HasPrefix(word, ".") {
			frac = word[1:]
		}
		if frac == "" {
			frac = "0"
		}
		for _, d := range frac {
			ws = append(ws, cardinal(int64(d-'0'))...)
		}
		ext(ws, nil)
	}
	if len(res) == 0 {
		return nil
	}
	for _, p := range res {
		if p == nil {
			return nil
		}
	}
	var b strings.Builder
	for i, p := range res {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(*p)
	}
	ps := b.String()
	switch suffix {
	case "s", "'s":
		return e.sSuffix(ps, nil, nil)
	case "ed", "'d":
		return edSuffix(&ps)
	case "ing":
		return ingSuffix(&ps)
	}
	return &ps
}

func firstRune(s string) rune {
	for _, r := range s {
		return r
	}
	return 0
}

func hasNonZero(rs []rune) bool {
	for _, r := range rs {
		if r != '0' {
			return true
		}
	}
	return false
}

func abs64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

func containsString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// ---- lexicon call -------------------------------------------------------

func (e *Engine) lexicon(text string, isHead bool, currency string, ctx *context) *string {
	w := strings.ReplaceAll(text, "‘", "'")
	w = strings.ReplaceAll(w, "’", "'")
	w = norm.NFKC.String(w)
	var stress *float64
	if w != strings.ToLower(w) {
		if w == strings.ToUpper(w) {
			stress = ptrF(2)
		} else {
			stress = ptrF(0.5)
		}
	}
	ps := e.getWord(w, stress, ctx)
	if ps != nil {
		if currency != "" {
			if u, ok := currencyUnits[firstRune(currency)]; ok {
				if s := e.stemS(u[0]+"s", nil, nil); s != nil {
					out := *ps + " " + *s
					return &out
				}
			}
		}
		return ps
	}
	if isNumber(w, isHead) {
		return e.getNumber(w, currency, isHead)
	}
	return nil
}

// ---- tokenizer ----------------------------------------------------------

type token struct {
	punct    bool
	text     string
	ps       string
	ws       string
	currency string
	subs     []string
}

var preChars = runeSet(`"“‘([{$£€`)
var postChars = runeSet(`"”)]}.,;:!?…`)

var abbrevs = runeSetWords("Adm Ala Apr Ariz Ark Aug Bros Calif Capt Cmdr Co Col Colo Conn Corp Dec Del Dr Feb Fla Ft Ga Gen Gov Ill Inc Ind " +
	"Jan Jr Jul Jun Kan Kans Ky La Lt Ltd Mar Mass Md Messrs Mfg Mich Minn Miss Mo Mont Mr Mrs Ms Mt Neb Nebr Nev Nov Oct Okla Ore Pa " +
	"Ph Prof Rep Rev Sen Sep Sept Sgt Sr St Tenn Va Wash Wis co vs etc")

func runeSetWords(s string) map[string]bool {
	m := make(map[string]bool)
	for _, w := range strings.Fields(s) {
		m[w] = true
	}
	return m
}

var (
	regexAbbrevMulti  = regexp.MustCompile(`^(?:[A-Za-z]{1,2}\.)+[A-Za-z]{1,2}$`)
	regexSingleLetter = regexp.MustCompile(`^[A-Za-z]$`)
	regexWhitespace   = regexp.MustCompile(`\s+`)
)

func (e *Engine) g2pOnce(text string) string {
	text = strings.TrimSpace(regexWhitespace.ReplaceAllString(text, " "))
	toks := e.tokenize(text)
	ctx := context{}
	for i := len(toks) - 1; i >= 0; i-- {
		t := &toks[i]
		if t.punct {
			ctx = nextContext(ctx, t.ps)
			ctx.nextIsWord = false
			continue
		}
		w := t.subs
		ps := make([]*string, len(w))
		left, right := 0, len(w)
		fallback := false
		for left < right {
			merged := strings.Join(w[left:right], "")
			cur := e.lexicon(merged, left == 0, t.currency, &ctx)
			if cur != nil {
				ps[left] = cur
				empty := ""
				for k := left + 1; k < right; k++ {
					ps[k] = &empty
				}
				ctx = nextContext(ctx, *cur)
				ctx.nextIsWord = true
				right = left
				left = 0
			} else if left+1 < right {
				left++
			} else {
				right--
				if ps[right] == nil {
					if allJunk(w[right]) {
						empty := ""
						ps[right] = &empty
					} else {
						fallback = true
						break
					}
				}
				left = 0
			}
		}
		if fallback {
			t.ps = e.spellFallback(strings.Join(w, ""))
		} else if len(w) == 1 {
			if ps[0] != nil {
				t.ps = *ps[0]
			}
		} else {
			t.ps = resolveGroup(w, ps)
		}
	}
	var b strings.Builder
	for _, t := range toks {
		b.WriteString(t.ps)
		b.WriteString(t.ws)
	}
	out := strings.TrimSpace(b.String())
	out = strings.ReplaceAll(out, "ɾ", "T")
	out = strings.ReplaceAll(out, "ʔ", "t")
	return out
}

func allJunk(s string) bool {
	for _, r := range s {
		if !subtokenJnk[r] {
			return false
		}
	}
	return true
}

func (e *Engine) spellFallback(word string) string {
	if e.wasm != nil {
		if s, ok := e.wasm.phonemize(word); ok {
			return s
		}
	}
	if e.espeak.available() {
		if s, ok := e.espeak.phonemize(word); ok {
			return s
		}
	}
	if p := e.getNNP(word); p != nil {
		return *p
	}
	// Last resort: spell every letter, including tagged entries.
	var out []rune
	for _, c := range word {
		if !unicode.IsLetter(c) {
			continue
		}
		v := e.gold.get(strings.ToUpper(string(c)))
		if v == nil {
			return ""
		}
		if v.tags != nil {
			out = append(out, []rune(*v.pick(nil))...)
		} else {
			out = append(out, []rune(v.text)...)
		}
	}
	return applyStress(string(out), ptrF(0))
}

func nextContext(ctx context, ps string) context {
	fv := ctx.futureVowel
	for _, c := range ps {
		if vowels[c] || consonants[c] || nonQuote[c] {
			if nonQuote[c] {
				fv = nil
			} else {
				b := vowels[c]
				fv = &b
			}
			break
		}
	}
	return context{futureVowel: fv}
}

func resolveGroup(subs []string, ps []*string) string {
	text := strings.Join(subs, "")
	kinds := make(map[int]bool)
	for _, c := range text {
		if subtokenJnk[c] {
			continue
		}
		if unicode.IsLetter(c) {
			kinds[0] = true
		} else if c >= '0' && c <= '9' {
			kinds[1] = true
		} else {
			kinds[2] = true
		}
	}
	prespace := strings.Contains(text, "/") || len(kinds) > 1
	if !prespace {
		var idx []groupItem
		for i, p := range ps {
			if p == nil || *p == "" {
				continue
			}
			prim := 0
			if strings.ContainsRune(*p, primary) {
				prim = 1
			}
			idx = append(idx, groupItem{prim, stressWeight(*p), i})
		}
		if len(idx) == 2 && len([]rune(subs[idx[0].i])) == 1 {
			ps[idx[1].i] = ptrS(applyStress(*ps[idx[1].i], ptrF(-0.5)))
		} else if len(idx) >= 2 {
			sum := 0
			for _, it := range idx {
				sum += it.prim
			}
			if sum > (len(idx)+1)/2 {
				sortItems(idx)
				half := len(idx) / 2
				for k := 0; k < half; k++ {
					i := idx[k].i
					ps[i] = ptrS(applyStress(*ps[i], ptrF(-0.5)))
				}
			}
		}
	}
	var b strings.Builder
	for i, p := range ps {
		if i > 0 && prespace && b.Len() > 0 && !strings.HasSuffix(b.String(), " ") && p != nil && *p != "" {
			b.WriteByte(' ')
		}
		if p != nil {
			b.WriteString(*p)
		}
	}
	return b.String()
}

type groupItem struct {
	prim int
	w    int
	i    int
}

func sortItems(items []groupItem) {
	for a := 1; a < len(items); a++ {
		for b := a; b > 0; b-- {
			x, y := items[b], items[b-1]
			if y.prim < x.prim || (y.prim == x.prim && (y.w < x.w || (y.w == x.w && y.i < x.i))) {
				break
			}
			items[b], items[b-1] = y, x
		}
	}
}

func (e *Engine) tokenize(text string) []token {
	var toks []token
	quoteOpen := false
	quote := func(open bool) string {
		quoteOpen = open
		if open {
			return "“"
		}
		return "”"
	}
	punct := func(t, ps string) {
		toks = append(toks, token{punct: true, text: t, ps: ps})
	}
	rs := []rune(text)
	for i := 0; i < len(rs); {
		// (\S+)(\s*)
		j := i
		for j < len(rs) && !unicode.IsSpace(rs[j]) {
			j++
		}
		c := rs[i:j]
		k := j
		for k < len(rs) && unicode.IsSpace(rs[k]) {
			k++
		}
		ws := string(rs[j:k])
		i = k
		if len(c) == 0 {
			continue
		}

		var tail [][2]string // ps is a sentinel "" meaning quote(false)
		tailQuote := make([]bool, 0)

		for len(c) > 0 && preChars[c[0]] {
			ch := c[0]
			if _, ok := currencyUnits[ch]; ok {
				toks = append(toks, token{punct: true, text: string(ch), currency: string(ch)})
			} else {
				var ps string
				switch ch {
				case '"':
					only := true
					for _, r := range c[1:] {
						if unicode.IsLetter(r) || unicode.IsDigit(r) {
							only = false
							break
						}
					}
					ps = quote(!(quoteOpen && only))
				case '“', '‘':
					ps = quote(true)
				case '(', '[', '{':
					ps = "("
				}
				punct(string(ch), ps)
			}
			c = c[1:]
		}

		for len(c) > 0 && postChars[c[len(c)-1]] {
			l := c[len(c)-1]
			prefix := string(c[:len(c)-1])
			if l == '.' {
				if regexAbbrevMulti.MatchString(prefix) || regexSingleLetter.MatchString(prefix) || abbrevs[prefix] {
					break
				}
				if strings.HasSuffix(string(c), "...") {
					tail = append([][2]string{{"...", "..."}}, tail...)
					tailQuote = append([]bool{false}, tailQuote...)
					c = c[:len(c)-3]
					continue
				}
			}
			switch {
			case l == '"' || l == '”':
				tail = append([][2]string{{string(l), ""}}, tail...)
				tailQuote = append([]bool{true}, tailQuote...)
			case strings.ContainsRune(")]}", l):
				tail = append([][2]string{{string(l), ")"}}, tail...)
				tailQuote = append([]bool{false}, tailQuote...)
			case puncts[l]:
				tail = append([][2]string{{string(l), string(l)}}, tail...)
				tailQuote = append([]bool{false}, tailQuote...)
			default:
				tail = append([][2]string{{string(l), ""}}, tail...)
				tailQuote = append([]bool{false}, tailQuote...)
			}
			c = c[:len(c)-1]
		}

		// Split on em/en dashes, ellipses and --.
		var pieces [][]rune
		var seps []string
		start := 0
		for p := 0; p < len(c); {
			var sep string
			if p+1 < len(c) && p+2 <= len(c) && c[p] == '-' && c[p+1] == '-' {
				sep = "--"
			} else if strings.ContainsRune("—–…", c[p]) {
				sep = string(c[p])
			}
			if sep == "" {
				p++
				continue
			}
			pieces = append(pieces, c[start:p])
			seps = append(seps, sep)
			p += len([]rune(sep))
			start = p
		}
		pieces = append(pieces, c[start:])
		for pi, piece := range pieces {
			if len(piece) == 0 {
				continue
			}
			s := string(piece)
			if s == "-" {
				punct(s, "—")
				continue
			}
			subs := subtokenize(s)
			var last *token
			if len(toks) > 0 {
				last = &toks[len(toks)-1]
			}
			if last == nil || last.punct || last.ws != "" {
				toks = append(toks, token{})
				last = &toks[len(toks)-1]
			}
			last.subs = append(last.subs, subs...)
			if pi < len(seps) {
				sp := "—"
				if seps[pi] == "…" {
					sp = "…"
				}
				punct(seps[pi], sp)
			}
		}

		for ti, t := range tail {
			if tailQuote[ti] {
				punct(t[0], quote(false))
			} else {
				punct(t[0], t[1])
			}
		}
		if ws != "" && len(toks) > 0 {
			toks[len(toks)-1].ws = " "
		}
	}

	// Currency: attach to the last number-like token after a currency sign.
	num := func(t *token) bool {
		if t == nil || t.punct || len(t.subs) == 0 {
			return false
		}
		if strings.ContainsAny(t.subs[0], "0123456789") {
			return true
		}
		switch strings.ToLower(strings.Join(t.subs, "")) {
		case "hundred", "thousand", "million", "billion", "trillion":
			return true
		}
		return false
	}
	for idx := 0; idx < len(toks); idx++ {
		if toks[idx].punct && toks[idx].currency != "" {
			j := idx + 1
			for j+1 < len(toks) && num(&toks[j+1]) {
				j++
			}
			if j < len(toks) && num(&toks[j]) {
				toks[j].currency = toks[idx].currency
			}
		}
	}
	return toks
}

// subtokenize is the Go equivalent of misaki-lite's SUBTOKEN regex. Go's
// regexp has no lookahead, so the alternatives are evaluated by hand.
func subtokenize(chunk string) []string {
	rs := []rune(chunk)
	var out []string
	n := len(rs)
	i := 0
	for i < n {
		r := rs[i]
		// ^['‘’]+
		if i == 0 && isApos(r) {
			j := i
			for j < n && isApos(rs[j]) {
				j++
			}
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}
		// \p{Lu}(?=\p{Lu}\p{Ll})
		if unicode.IsUpper(r) && i+2 < n && unicode.IsUpper(rs[i+1]) && unicode.IsLower(rs[i+2]) {
			out = append(out, string(r))
			i++
			continue
		}
		// numbers
		if (r >= '0' && r <= '9') || (r == '-' && i == 0 && i+1 < n && rs[i+1] >= '0' && rs[i+1] <= '9') {
			j := i
			if rs[j] == '-' {
				j++
			}
			for j < n {
				if rs[j] >= '0' && rs[j] <= '9' {
					j++
					continue
				}
				if (rs[j] == ',' || rs[j] == '.') && j+1 < n && rs[j+1] >= '0' && rs[j+1] <= '9' {
					j += 2
					continue
				}
				break
			}
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}
		if r == '-' || r == '_' {
			j := i
			for j < n && (rs[j] == '-' || rs[j] == '_') {
				j++
			}
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}
		if isApos(r) {
			j := i
			for j < n && isApos(rs[j]) {
				j++
			}
			if j-i >= 2 || j == n {
				out = append(out, string(rs[i:j]))
				i = j
				continue
			}
			out = append(out, string(r))
			i++
			continue
		}
		if unicode.IsLetter(r) {
			// shortest letter run ending in lowercase before uppercase
			target := -1
			for j := i; j < n; {
				c := rs[j]
				if unicode.IsLetter(c) {
					if unicode.IsLower(c) && j+1 < n && unicode.IsUpper(rs[j+1]) {
						target = j
						break
					}
					j++
					continue
				}
				if isApos(c) && j+2 < n && unicode.IsLetter(rs[j+1]) {
					j += 2
					continue
				}
				break
			}
			if target >= 0 {
				out = append(out, string(rs[i:target+1]))
				i = target + 1
				continue
			}
			j := i
			for j < n {
				if unicode.IsLetter(rs[j]) {
					j++
					continue
				}
				if isApos(rs[j]) && j+2 < n && unicode.IsLetter(rs[j+1]) {
					j += 2
					continue
				}
				break
			}
			if j > i {
				out = append(out, string(rs[i:j]))
				i = j
				continue
			}
		}
		out = append(out, string(r))
		i++
	}
	return out
}

func isApos(r rune) bool { return r == '\'' || r == '‘' || r == '’' }
