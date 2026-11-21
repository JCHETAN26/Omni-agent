package sink

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/JCHETAN26/Omni-agent/ingestion/internal/chunker"
)

func TestBulkUsesChunkIDsAndReportsItemErrors(t *testing.T) {
	var body, query string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body, query = string(b), r.URL.RawQuery
		w.Write([]byte(`{"errors":true,"items":[
			{"index":{"_id":"c1","status":201}},
			{"index":{"_id":"c2","status":400,"error":{"type":"mapper_parsing_exception"}}},
			{"index":{"_id":"c3","status":429,"error":{"type":"es_rejected_execution_exception"}}}]}`))
	}))
	defer srv.Close()
	es := NewElasticsearch(srv.URL, "", "idx")
	items, err := es.Bulk(context.Background(), []chunker.Chunk{{ChunkID: "c1"}, {ChunkID: "c2"}, {ChunkID: "c3"}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(body, `"_id":"c1"`) || query != "refresh=wait_for" {
		t.Fatalf("body=%s query=%s", body, query)
	}
	if len(items) != 2 || items[0].Err.Retryable || !items[1].Err.Retryable {
		t.Fatalf("items %+v", items)
	}
}

func TestRetryClassification(t *testing.T) {
	for code, want := range map[int]bool{500: true, 503: true, 429: true, 400: false, 404: false} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(code) }))
		err := NewQdrant(srv.URL, "", "c").Upsert(context.Background(), []chunker.Chunk{{ChunkID: "x"}}, [][]float32{{1}})
		srv.Close()
		if IsRetryable(err) != want {
			t.Errorf("HTTP %d: retryable=%v, want %v (%v)", code, IsRetryable(err), want, err)
		}
	}
	err := NewQdrant("http://127.0.0.1:1", "", "c").Upsert(context.Background(), []chunker.Chunk{{ChunkID: "x"}}, [][]float32{{1}})
	if !IsRetryable(err) {
		t.Errorf("connection refused must be retryable: %v", err)
	}
}
