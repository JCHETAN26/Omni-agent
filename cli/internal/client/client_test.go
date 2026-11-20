package client

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func sse(events ...map[string]any) string {
	var b strings.Builder
	for i, ev := range events {
		raw, _ := json.Marshal(ev)
		fmt.Fprintf(&b, "id: %d\ndata: %s\n\n", i+1, raw)
	}
	return b.String()
}

func end(text string, tokens int) map[string]any {
	sum := sha256.Sum256([]byte(text))
	return map[string]any{
		"type": "response.end", "tokens": tokens, "sha256": hex.EncodeToString(sum[:]),
		"outcome": "answered", "tool": "knowledge_base",
	}
}

func TestAskChecksIntegrityAndSendsThread(t *testing.T) {
	text := "Read the runbook. [1]"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/sse/agent" || r.URL.Query().Get("q") != "Which one?" ||
			r.URL.Query().Get("thread_id") != "t-1" {
			http.Error(w, "bad request", 400)
			return
		}
		w.Header().Set("X-Thread-Id", "t-1")
		fmt.Fprint(w, sse(
			map[string]any{"type": "response.sources", "sources": []any{map[string]any{"doc_id": "runbooks/a.md"}}},
			map[string]any{"type": "response.token", "index": 0, "text": "Read the runbook. "},
			map[string]any{"type": "response.token", "index": 1, "text": "[1]"},
			end(text, 2),
		))
	}))
	defer srv.Close()

	ans, err := New(srv.URL).Ask(context.Background(), "Which one?", "t-1")
	if err != nil {
		t.Fatal(err)
	}
	if ans.Text != text || ans.ThreadID != "t-1" || ans.Sources[0]["doc_id"] != "runbooks/a.md" {
		t.Fatalf("unexpected answer %+v", ans)
	}
	if ans.TTFTms == nil || ans.TotalMs < *ans.TTFTms {
		t.Fatalf("timings inconsistent: %+v", ans)
	}
}

func TestAskRejectsBadStreams(t *testing.T) {
	cases := map[string]struct{ body, want string }{
		"token order": {sse(map[string]any{"type": "response.token", "index": 1, "text": "x"}), "token sequence"},
		"checksum":    {sse(map[string]any{"type": "response.end", "tokens": 0, "sha256": "bad"}), "integrity"},
		"truncated":   {"", "without a complete answer"},
		"event order": {"id: 2\ndata: {}\n\n", "event sequence"},
		"error event": {sse(map[string]any{"type": "response.error", "error": "boom"}), "boom"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				fmt.Fprint(w, tc.body)
			}))
			defer srv.Close()
			_, err := New(srv.URL).Ask(context.Background(), "q", "")
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("want error containing %q, got %v", tc.want, err)
			}
		})
	}
}

func TestValidateServer(t *testing.T) {
	for _, ok := range []string{"http://127.0.0.1:8000", "https://omni.example.com/"} {
		if _, err := ValidateServer(ok); err != nil {
			t.Errorf("%s rejected: %v", ok, err)
		}
	}
	for _, bad := range []string{"ftp://x", "http://", "http://u:p@host", "http://h?q=1", "http://h#f", "not a url"} {
		if _, err := ValidateServer(bad); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
}

func TestCollectRunbooks(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "runbooks")
	must := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	must(os.MkdirAll(filepath.Join(dir, "storage"), 0o755))
	must(os.WriteFile(filepath.Join(dir, "storage", "eio.md"), []byte("intro\n# EIO-5 errors\nbody"), 0o644))
	must(os.WriteFile(filepath.Join(dir, "plain.md"), []byte("no heading"), 0o644))
	must(os.WriteFile(filepath.Join(dir, "empty.md"), []byte("  \n"), 0o644))
	must(os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("ignored"), 0o644))
	must(os.Symlink(filepath.Join(dir, "plain.md"), filepath.Join(dir, "link.md")))

	docs, err := CollectRunbooks(dir)
	must(err)
	if len(docs) != 2 {
		t.Fatalf("want 2 docs, got %+v", docs)
	}
	if docs[0].DocID != "runbooks/plain.md" || docs[0].Title != "plain" {
		t.Errorf("bad first doc %+v", docs[0])
	}
	if docs[1].DocID != "runbooks/storage/eio.md" || docs[1].Title != "EIO-5 errors" {
		t.Errorf("bad second doc %+v", docs[1])
	}
	if _, err := CollectRunbooks(t.TempDir()); err == nil {
		t.Error("empty dir accepted")
	}
}
