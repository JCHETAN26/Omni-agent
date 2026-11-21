package chunker

import (
	"encoding/json"
	"os"
	"testing"
)

type goldenCase struct {
	Doc struct {
		DocID string `json:"doc_id"`
		Title string `json:"title"`
		Text  string `json:"text"`
	} `json:"doc"`
	Size    int     `json:"size"`
	Overlap int     `json:"overlap"`
	Chunks  []Chunk `json:"chunks"`
}

// TestMatchesPythonChunker replays every case generated from
// backend/ingestion/chunker.py (scripts/gen_ingestion_golden.py).
func TestMatchesPythonChunker(t *testing.T) {
	raw, err := os.ReadFile("testdata/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []goldenCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	total := 0
	for _, c := range cases {
		got, err := Split(Document{c.Doc.DocID, c.Doc.Title, c.Doc.Text}, c.Size, c.Overlap)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(c.Chunks) {
			t.Errorf("%s (%d/%d): %d chunks, want %d", c.Doc.DocID, c.Size, c.Overlap, len(got), len(c.Chunks))
			continue
		}
		for i := range got {
			if got[i] != c.Chunks[i] {
				t.Errorf("%s (%d/%d) chunk %d:\n got %#v\nwant %#v", c.Doc.DocID, c.Size, c.Overlap, i, got[i], c.Chunks[i])
				break
			}
		}
		total += len(got)
	}
	if total < 1000 {
		t.Fatalf("only %d golden chunks checked", total)
	}
}

func TestChunkIDIsUUIDv5(t *testing.T) {
	// Version nibble 5, variant 10xx.
	id := ChunkID("doc-1", 0)
	if len(id) != 36 || id[14] != '5' {
		t.Fatalf("not a v5 UUID: %s", id)
	}
	if ChunkID("doc-1", 0) != id || ChunkID("doc-1", 1) == id {
		t.Fatal("chunk IDs must be deterministic and position-specific")
	}
}

func TestOverlapMustBeSmaller(t *testing.T) {
	if _, err := Split(Document{DocID: "d", Text: "x"}, 10, 10); err == nil {
		t.Fatal("expected an error")
	}
}
