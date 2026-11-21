package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	pb "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
)

// BM25 queries Elasticsearch directly: the degraded path when no gateway can
// run hybrid search (typically Triton or Qdrant is down, which takes every
// gateway out). Same index and query shape as the gateway's sparse branch.
type BM25 struct {
	URL, Index, APIKey string
	HTTP               *http.Client
}

func NewBM25(url, index, apiKey string) *BM25 {
	return &BM25{URL: url, Index: index, APIKey: apiKey, HTTP: &http.Client{Timeout: 5 * time.Second}}
}

func (b *BM25) Search(ctx context.Context, query string, k int32) ([]*pb.Hit, error) {
	body, _ := json.Marshal(map[string]any{
		"size": k,
		"query": map[string]any{"multi_match": map[string]any{
			"query": query, "fields": []string{"text", "section_heading^0.5", "document_title^0.3"},
		}},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.URL+"/"+b.Index+"/_search", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if b.APIKey != "" {
		req.Header.Set("Authorization", "ApiKey "+b.APIKey)
	}
	resp, err := b.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("elasticsearch HTTP %d: %.200s", resp.StatusCode, raw)
	}
	var r struct {
		Hits struct {
			Hits []struct {
				Score  float64 `json:"_score"`
				Source struct {
					ChunkID        string `json:"chunk_id"`
					DocID          string `json:"doc_id"`
					DocumentTitle  string `json:"document_title"`
					SectionHeading string `json:"section_heading"`
					Text           string `json:"text"`
				} `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return nil, err
	}
	hits := make([]*pb.Hit, len(r.Hits.Hits))
	for i, h := range r.Hits.Hits {
		s := h.Source
		hits[i] = &pb.Hit{ChunkId: s.ChunkID, DocId: s.DocID, DocumentTitle: s.DocumentTitle,
			SectionHeading: s.SectionHeading, Text: s.Text, Score: h.Score}
	}
	return hits, nil
}
