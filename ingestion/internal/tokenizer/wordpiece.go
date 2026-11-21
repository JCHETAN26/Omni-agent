// Package tokenizer is the BERT uncased WordPiece tokenizer used by
// sentence-transformers/all-MiniLM-L6-v2, matching Hugging Face `tokenizers`
// (BertNormalizer + BertPreTokenizer + WordPiece + [CLS]/[SEP]) so the Go
// indexer sends Triton the same input_ids the Python gateway does.
// testdata/golden.json (from the Python tokenizer) pins that equivalence.
package tokenizer

import (
	"bufio"
	"bytes"
	_ "embed"
	"fmt"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

//go:embed vocab.txt
var vocabFile []byte

const maxCharsPerWord = 100

// Tokenizer maps text to WordPiece IDs.
type Tokenizer struct {
	vocab              map[string]int64
	cls, sep, unk, pad int64
}

// MiniLM returns the all-MiniLM-L6-v2 tokenizer (vocabulary embedded in the binary).
func MiniLM() (*Tokenizer, error) { return New(vocabFile) }

// New builds a tokenizer from a vocab.txt (one token per line, ID = line number).
func New(vocabTxt []byte) (*Tokenizer, error) {
	t := &Tokenizer{vocab: make(map[string]int64, 32000)}
	sc := bufio.NewScanner(bytes.NewReader(vocabTxt))
	var id int64
	for sc.Scan() {
		t.vocab[strings.TrimRight(sc.Text(), "\r")] = id
		id++
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	for name, dst := range map[string]*int64{"[CLS]": &t.cls, "[SEP]": &t.sep, "[UNK]": &t.unk, "[PAD]": &t.pad} {
		v, ok := t.vocab[name]
		if !ok {
			return nil, fmt.Errorf("vocab has no %s", name)
		}
		*dst = v
	}
	return t, nil
}

// PadID is the [PAD] token ID.
func (t *Tokenizer) PadID() int64 { return t.pad }

// Encode returns [CLS] tokens... [SEP], truncated to maxLen IDs in total
// (tokenizer(text, truncation=True, max_length=maxLen)["input_ids"]).
func (t *Tokenizer) Encode(text string, maxLen int) []int64 {
	ids := []int64{t.cls}
	budget := maxLen - 2
	for _, word := range preTokenize(normalize(text)) {
		for _, id := range t.wordPiece(word) {
			if len(ids)-1 >= budget {
				return append(ids, t.sep)
			}
			ids = append(ids, id)
		}
	}
	return append(ids, t.sep)
}

func (t *Tokenizer) wordPiece(word string) []int64 {
	chars := []rune(word)
	if len(chars) > maxCharsPerWord {
		return []int64{t.unk}
	}
	var out []int64
	for start := 0; start < len(chars); {
		end := len(chars)
		var id int64 = -1
		for ; end > start; end-- {
			piece := string(chars[start:end])
			if start > 0 {
				piece = "##" + piece
			}
			if v, ok := t.vocab[piece]; ok {
				id = v
				break
			}
		}
		if id < 0 {
			return []int64{t.unk}
		}
		out = append(out, id)
		start = end
	}
	return out
}

// isWhitespace / isControl follow tokenizers' BertNormalizer.
func isWhitespace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\r':
		return true
	}
	return unicode.Is(unicode.Zs, r)
}

func isControl(r rune) bool {
	switch r {
	case '\t', '\n', '\r':
		return false
	}
	return unicode.In(r, unicode.Cc, unicode.Cf, unicode.Co, unicode.Cs) || !isAssigned(r)
}

func isAssigned(r rune) bool {
	for _, table := range unicode.Categories {
		if unicode.Is(table, r) {
			return true
		}
	}
	return false
}

func isChinese(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) || (r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) || (r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) || (r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) || (r >= 0x2F800 && r <= 0x2FA1F)
}

// normalize: clean text, pad CJK with spaces, strip accents (NFD, drop Mn), lowercase.
func normalize(text string) string {
	var b strings.Builder
	for _, r := range text {
		switch {
		case r == 0 || r == 0xFFFD || isControl(r):
		case isWhitespace(r):
			b.WriteByte(' ')
		case isChinese(r):
			b.WriteByte(' ')
			b.WriteRune(r)
			b.WriteByte(' ')
		default:
			b.WriteRune(r)
		}
	}
	var out strings.Builder
	for _, r := range norm.NFD.String(b.String()) {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		out.WriteString(strings.ToLower(string(r)))
	}
	return out.String()
}

func isPunct(r rune) bool {
	if r < 128 {
		return (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126)
	}
	return unicode.In(r, unicode.Pc, unicode.Pd, unicode.Pe, unicode.Pf, unicode.Pi, unicode.Po, unicode.Ps)
}

// preTokenize splits on whitespace (dropped) and punctuation (kept, one char each).
func preTokenize(text string) []string {
	var words []string
	var cur []rune
	flush := func() {
		if len(cur) > 0 {
			words = append(words, string(cur))
			cur = cur[:0]
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			flush()
		case isPunct(r):
			flush()
			words = append(words, string(r))
		default:
			cur = append(cur, r)
		}
	}
	flush()
	return words
}
