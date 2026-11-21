// Package chunker is a Go port of backend/ingestion/chunker.py.
//
// It must produce byte-identical chunks and chunk IDs to the Python chunker:
// the chunk ID (UUIDv5 of "doc_id:position") is the Elasticsearch _id and the
// Qdrant point ID, so the two pipelines (and replays of either) overwrite the
// same records instead of creating duplicates. testdata/golden.json, generated
// from the Python implementation, pins that equivalence.
//
// Python semantics reproduced here: lengths count code points (runes), `\s`,
// str.strip() and str.isspace() use Unicode whitespace, and str.splitlines()
// breaks on every Unicode line boundary.
package chunker

import (
	"crypto/sha1"
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Document is the unit of ingestion.
type Document struct {
	DocID string
	Title string
	Text  string
}

// Chunk mirrors backend.core.models.Chunk.
type Chunk struct {
	ChunkID        string `json:"chunk_id"`
	DocID          string `json:"doc_id"`
	DocumentTitle  string `json:"document_title"`
	SectionHeading string `json:"section_heading"`
	Position       int    `json:"position"`
	Text           string `json:"text"`
}

// Python's str.isspace() / regex \s for str patterns.
func isSpace(r rune) bool {
	return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f)
}

// pyWS is a regexp class equivalent to Python's Unicode \s.
const pyWS = `[\t\n\x0b\x0c\r \x1c-\x1f\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}]`

var (
	headingRE = regexp.MustCompile(`^(#{1,6})` + pyWS + `+(.+?)` + pyWS + `*#*` + pyWS + `*$`)
	fenceRE   = regexp.MustCompile(`^` + pyWS + "*(```|~~~)")
	paraRE    = regexp.MustCompile(`\n` + pyWS + `*\n`)
)

func strip(s string) string { return strings.TrimFunc(s, isSpace) }

// splitLines is str.splitlines() (no keepends).
func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		switch r {
		case '\n', '\r', '\x0b', '\x0c', '\x1c', '\x1d', '\x1e', '\u0085', ' ', ' ':
			lines = append(lines, s[start:i])
			next := i + size
			if r == '\r' && next < len(s) && s[next] == '\n' {
				next++
			}
			start, i = next, next
			continue
		}
		i += size
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

type section struct{ heading, text string }

type level struct {
	n     int
	title string
}

func splitSections(doc Document) []section {
	var sections []section
	var path []level
	var buf []string
	inFence := false
	flush := func() {
		body := strip(strings.Join(buf, "\n"))
		if body != "" {
			titles := make([]string, len(path))
			for i, l := range path {
				titles[i] = l.title
			}
			heading := strings.Join(titles, " > ")
			if heading == "" {
				heading = doc.Title
			}
			sections = append(sections, section{heading, body})
		}
		buf = buf[:0]
	}
	for _, line := range splitLines(doc.Text) {
		if fenceRE.MatchString(line) {
			inFence = !inFence
		}
		var m []string
		if !inFence {
			m = headingRE.FindStringSubmatch(line)
		}
		if m != nil {
			flush()
			n, title := len(m[1]), strip(m[2])
			for len(path) > 0 && path[len(path)-1].n >= n {
				path = path[:len(path)-1]
			}
			path = append(path, level{n, title})
		} else {
			buf = append(buf, line)
		}
	}
	flush()
	return sections
}

func runeLen(s string) int { return utf8.RuneCountInString(s) }

func hardSplit(text string, size int) []string {
	r := []rune(text)
	var out []string
	for i := 0; i < len(r); i += size {
		out = append(out, string(r[i:min(i+size, len(r))]))
	}
	return out
}

// splitSentences is re.split(r"(?<=[.!?])\s+", text).
func splitSentences(text string) []string {
	r := []rune(text)
	var out []string
	start := 0
	for i := 1; i < len(r); i++ {
		if !isSpace(r[i]) || (r[i-1] != '.' && r[i-1] != '!' && r[i-1] != '?') {
			continue
		}
		j := i
		for j < len(r) && isSpace(r[j]) {
			j++
		}
		out = append(out, string(r[start:i]))
		start = j
		i = j - 1
	}
	return append(out, string(r[start:]))
}

func units(sectionText string, size int) []string {
	var out []string
	for _, para := range paraRE.Split(sectionText, -1) {
		para = strip(para)
		if para == "" {
			continue
		}
		if runeLen(para) <= size {
			out = append(out, para)
			continue
		}
		for _, sent := range splitSentences(para) {
			if runeLen(sent) <= size {
				out = append(out, sent)
			} else {
				out = append(out, hardSplit(sent, size)...)
			}
		}
	}
	return out
}

// tail returns the last ~n characters of text, starting on a word boundary.
func tail(text string, n int) string {
	r := []rune(text)
	if n <= 0 || len(r) <= n {
		if n > 0 {
			return text
		}
		return ""
	}
	cut := r[len(r)-n:]
	space := -1
	for i, c := range cut {
		if c == ' ' {
			space = i
			break
		}
	}
	if space >= 0 && space < len(cut)-1 {
		return string(cut[space+1:])
	}
	return string(cut)
}

func pack(us []string, size, overlap int) []string {
	var chunks []string
	current := ""
	for _, unit := range us {
		candidate := unit
		if current != "" {
			candidate = current + "\n\n" + unit
		}
		if runeLen(candidate) <= size {
			current = candidate
			continue
		}
		if current != "" {
			chunks = append(chunks, current)
			carry := tail(current, overlap)
			if carry != "" && runeLen(carry)+runeLen(unit)+2 <= size {
				current = carry + "\n\n" + unit
			} else {
				current = unit
			}
		} else {
			current = unit
		}
	}
	if current != "" {
		chunks = append(chunks, current)
	}
	return chunks
}

// namespace is _CHUNK_NS in chunker.py.
var namespace = [16]byte{
	0x6f, 0x1c, 0x2a, 0x8e, 0x3b, 0x7d, 0x4c, 0x55,
	0x9a, 0x0e, 0x0f, 0x6d, 0x2b, 0x9e, 0x7a, 0x11,
}

// ChunkID is uuid.uuid5(_CHUNK_NS, f"{doc_id}:{position}").
func ChunkID(docID string, position int) string {
	h := sha1.New()
	h.Write(namespace[:])
	fmt.Fprintf(h, "%s:%d", docID, position)
	var u [16]byte
	copy(u[:], h.Sum(nil))
	u[6] = (u[6] & 0x0f) | 0x50
	u[8] = (u[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", u[0:4], u[4:6], u[6:8], u[8:10], u[10:16])
}

// Split chunks a document exactly like chunk_document(doc, size, overlap).
func Split(doc Document, size, overlap int) ([]Chunk, error) {
	if overlap >= size {
		return nil, fmt.Errorf("chunk overlap (%d) must be smaller than chunk size (%d)", overlap, size)
	}
	var out []Chunk
	for _, s := range splitSections(doc) {
		for _, text := range pack(units(s.text, size), size, overlap) {
			pos := len(out)
			out = append(out, Chunk{
				ChunkID:        ChunkID(doc.DocID, pos),
				DocID:          doc.DocID,
				DocumentTitle:  doc.Title,
				SectionHeading: s.heading,
				Position:       pos,
				Text:           text,
			})
		}
	}
	return out, nil
}
