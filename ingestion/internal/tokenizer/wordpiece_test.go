package tokenizer

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

// TestMatchesHuggingFace replays input_ids produced by the Python
// all-MiniLM-L6-v2 tokenizer (scripts/gen_ingestion_golden.py).
func TestMatchesHuggingFace(t *testing.T) {
	raw, err := os.ReadFile("testdata/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text     string  `json:"text"`
		InputIDs []int64 `json:"input_ids"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	tok, err := MiniLM()
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range cases {
		if got := tok.Encode(c.Text, 256); !slices.Equal(got, c.InputIDs) {
			t.Errorf("case %d %q:\n got %v\nwant %v", i, trunc(c.Text), got, c.InputIDs)
		}
	}
}

func trunc(s string) string {
	if r := []rune(s); len(r) > 60 {
		return string(r[:60]) + "..."
	}
	return s
}
