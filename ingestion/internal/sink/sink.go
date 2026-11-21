// Package sink writes chunks to Elasticsearch (BM25) and Qdrant (dense), with
// the same index mapping and collection layout as backend/core/stores.py.
// Record IDs are the deterministic chunk IDs, so every write is an idempotent
// upsert: replaying a Kafka record overwrites, never duplicates.
package sink

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/JCHETAN26/Omni-agent/ingestion/internal/chunker"
)

// Error is a failed store call. Retryable errors (connection failures, 429,
// 5xx) mean "try again later"; the rest are permanent for that input.
type Error struct {
	Store     string
	Status    int
	Retryable bool
	Err       error
}

func (e *Error) Error() string {
	return fmt.Sprintf("%s (HTTP %d): %v", e.Store, e.Status, e.Err)
}
func (e *Error) Unwrap() error { return e.Err }

// IsRetryable reports whether err is an outage worth retrying unchanged
// (any error in the chain implementing IsRetryable, or a timeout).
func IsRetryable(err error) bool {
	var r interface{ IsRetryable() bool }
	if errors.As(err, &r) {
		return r.IsRetryable()
	}
	return errors.Is(err, context.DeadlineExceeded)
}

func (e *Error) IsRetryable() bool { return e.Retryable }

func retryableStatus(code int) bool {
	return code == http.StatusTooManyRequests || code >= 500 || code == http.StatusRequestTimeout
}

type httpStore struct {
	name   string
	base   string
	header http.Header
	client *http.Client
}

func (s *httpStore) do(ctx context.Context, method, path string, body []byte, contentType string) (int, []byte, error) {
	req, err := http.NewRequestWithContext(ctx, method, s.base+path, bytes.NewReader(body))
	if err != nil {
		return 0, nil, &Error{Store: s.name, Err: err}
	}
	for k, v := range s.header {
		req.Header[k] = v
	}
	if body != nil {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return 0, nil, &Error{Store: s.name, Retryable: true, Err: err}
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, nil, &Error{Store: s.name, Status: resp.StatusCode, Retryable: true, Err: err}
	}
	return resp.StatusCode, raw, nil
}

func (s *httpStore) fail(status int, raw []byte) error {
	msg := bytes.TrimSpace(raw)
	if len(msg) > 512 {
		msg = msg[:512]
	}
	return &Error{Store: s.name, Status: status, Retryable: retryableStatus(status), Err: fmt.Errorf("%s", msg)}
}

// Elasticsearch is the BM25 index.
type Elasticsearch struct {
	httpStore
	Index string
	// Refresh=wait_for makes a bulk return only once its chunks are searchable,
	// the same guarantee the Python pipeline gives.
	Refresh string
}

func NewElasticsearch(url, apiKey, index string) *Elasticsearch {
	h := http.Header{}
	if apiKey != "" {
		h.Set("Authorization", "ApiKey "+apiKey)
	}
	return &Elasticsearch{
		httpStore: httpStore{"elasticsearch", url, h, &http.Client{Timeout: 120 * time.Second}},
		Index:     index,
		Refresh:   "wait_for",
	}
}

// EnsureIndex creates the index with backend/core/stores.py's mapping if missing.
func (e *Elasticsearch) EnsureIndex(ctx context.Context) error {
	status, raw, err := e.do(ctx, http.MethodHead, "/"+e.Index, nil, "")
	if err != nil {
		return err
	}
	if status == http.StatusOK {
		return nil
	}
	if status != http.StatusNotFound {
		return e.fail(status, raw)
	}
	text := map[string]string{"type": "text", "similarity": "omni_bm25"}
	body, _ := json.Marshal(map[string]any{
		"settings": map[string]any{
			"number_of_shards": 1,
			"similarity":       map[string]any{"omni_bm25": map[string]any{"type": "BM25", "k1": 1.2, "b": 0.75}},
		},
		"mappings": map[string]any{"properties": map[string]any{
			"chunk_id":        map[string]string{"type": "keyword"},
			"doc_id":          map[string]string{"type": "keyword"},
			"document_title":  text,
			"section_heading": text,
			"position":        map[string]string{"type": "integer"},
			"text":            text,
		}},
	})
	status, raw, err = e.do(ctx, http.MethodPut, "/"+e.Index, body, "application/json")
	if err != nil {
		return err
	}
	if status >= 300 && !bytes.Contains(raw, []byte("resource_already_exists_exception")) {
		return e.fail(status, raw)
	}
	return nil
}

// ItemError is a chunk Elasticsearch rejected inside an otherwise successful bulk.
type ItemError struct {
	ChunkID string
	Err     *Error
}

// Bulk indexes chunks by chunk ID. A transport/HTTP failure returns err; chunks
// rejected individually come back as item errors.
func (e *Elasticsearch) Bulk(ctx context.Context, chunks []chunker.Chunk) ([]ItemError, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, c := range chunks {
		_ = enc.Encode(map[string]any{"index": map[string]string{"_index": e.Index, "_id": c.ChunkID}})
		_ = enc.Encode(c)
	}
	path := "/_bulk"
	if e.Refresh != "" {
		path += "?refresh=" + e.Refresh
	}
	status, raw, err := e.do(ctx, http.MethodPost, path, buf.Bytes(), "application/x-ndjson")
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, e.fail(status, raw)
	}
	var resp struct {
		Errors bool `json:"errors"`
		Items  []map[string]struct {
			ID     string          `json:"_id"`
			Status int             `json:"status"`
			Error  json.RawMessage `json:"error"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, &Error{Store: e.name, Status: status, Retryable: true, Err: err}
	}
	if len(resp.Items) != len(chunks) {
		return nil, &Error{Store: e.name, Status: status, Retryable: true,
			Err: fmt.Errorf("bulk returned %d items for %d chunks", len(resp.Items), len(chunks))}
	}
	if !resp.Errors {
		return nil, nil
	}
	var failed []ItemError
	for _, item := range resp.Items {
		for _, r := range item {
			if r.Status >= 300 {
				failed = append(failed, ItemError{r.ID, &Error{
					Store: e.name, Status: r.Status, Retryable: retryableStatus(r.Status),
					Err: fmt.Errorf("%s", r.Error),
				}})
			}
		}
	}
	return failed, nil
}

// Qdrant is the dense store.
type Qdrant struct {
	httpStore
	Collection   string
	Quantization string // "int8" or "" (matches QDRANT_QUANTIZATION in the gateway)
	UpsertBatch  int
}

func NewQdrant(url, apiKey, collection string) *Qdrant {
	h := http.Header{}
	if apiKey != "" {
		h.Set("api-key", apiKey)
	}
	return &Qdrant{
		httpStore:   httpStore{"qdrant", url, h, &http.Client{Timeout: 120 * time.Second}},
		Collection:  collection,
		UpsertBatch: 512,
	}
}

// EnsureCollection creates the cosine collection (and a doc_id keyword index) if missing.
func (q *Qdrant) EnsureCollection(ctx context.Context, dim int) error {
	path := "/collections/" + q.Collection
	status, raw, err := q.do(ctx, http.MethodGet, path, nil, "")
	if err != nil {
		return err
	}
	if status == http.StatusNotFound {
		spec := map[string]any{"vectors": map[string]any{"size": dim, "distance": "Cosine"}}
		if q.Quantization == "int8" {
			spec["quantization_config"] = map[string]any{
				"scalar": map[string]any{"type": "int8", "quantile": 0.99, "always_ram": true},
			}
		}
		body, _ := json.Marshal(spec)
		status, raw, err = q.do(ctx, http.MethodPut, path, body, "application/json")
		if err != nil {
			return err
		}
		if status >= 300 && !bytes.Contains(raw, []byte("already exists")) {
			return q.fail(status, raw)
		}
	} else if status != http.StatusOK {
		return q.fail(status, raw)
	}
	body := []byte(`{"field_name":"doc_id","field_schema":"keyword"}`)
	status, raw, err = q.do(ctx, http.MethodPut, path+"/index?wait=true", body, "application/json")
	if err != nil {
		return err
	}
	if status >= 300 {
		return q.fail(status, raw)
	}
	return nil
}

type point struct {
	ID      string        `json:"id"`
	Vector  []float32     `json:"vector"`
	Payload chunker.Chunk `json:"payload"`
}

// Upsert writes points with wait=true: on success they are applied and searchable.
func (q *Qdrant) Upsert(ctx context.Context, chunks []chunker.Chunk, vectors [][]float32) error {
	if len(chunks) != len(vectors) {
		return &Error{Store: q.name, Err: fmt.Errorf("%d chunks, %d vectors", len(chunks), len(vectors))}
	}
	for start := 0; start < len(chunks); start += q.UpsertBatch {
		end := min(start+q.UpsertBatch, len(chunks))
		points := make([]point, 0, end-start)
		for i := start; i < end; i++ {
			points = append(points, point{chunks[i].ChunkID, vectors[i], chunks[i]})
		}
		body, err := json.Marshal(map[string]any{"points": points})
		if err != nil {
			return &Error{Store: q.name, Err: err}
		}
		status, raw, err := q.do(ctx, http.MethodPut,
			"/collections/"+q.Collection+"/points?wait=true", body, "application/json")
		if err != nil {
			return err
		}
		if status != http.StatusOK {
			return q.fail(status, raw)
		}
	}
	return nil
}
