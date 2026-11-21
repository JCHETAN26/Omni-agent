// Package embed calls the minilm_embedder model on Triton (KServe v2 HTTP),
// tokenizing in Go exactly like the Python gateway: WordPiece, max 256 tokens,
// padded to a length bucket (32/64/128/256) so Triton's dynamic batcher can
// merge requests of equal shape.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/JCHETAN26/Omni-agent/ingestion/internal/tokenizer"
)

const (
	MaxTokens = 256
	Dim       = 384
)

var buckets = []int{32, 64, 128, 256}

// BucketLength mirrors model_server.inference_client.bucket_length.
func BucketLength(n int) int {
	for _, b := range buckets {
		if n <= b {
			return b
		}
	}
	return buckets[len(buckets)-1]
}

// Error is a failed Triton call; Retryable distinguishes outages from bad requests.
type Error struct {
	Status    int
	Retryable bool
	Err       error
}

func (e *Error) Error() string     { return fmt.Sprintf("triton embed (HTTP %d): %v", e.Status, e.Err) }
func (e *Error) Unwrap() error     { return e.Err }
func (e *Error) IsRetryable() bool { return e.Retryable }

// Client embeds texts with Triton.
type Client struct {
	BaseURL   string // e.g. http://omniagent-triton:8000
	Model     string // minilm_embedder
	MaxBatch  int    // texts per request (model max_batch_size)
	HTTP      *http.Client
	tokenizer *tokenizer.Tokenizer
}

func New(baseURL, model string, maxBatch int) (*Client, error) {
	tok, err := tokenizer.MiniLM()
	if err != nil {
		return nil, err
	}
	return &Client{
		BaseURL: baseURL, Model: model, MaxBatch: maxBatch,
		HTTP:      &http.Client{Timeout: 60 * time.Second},
		tokenizer: tok,
	}, nil
}

type tensor struct {
	Name     string  `json:"name"`
	Shape    []int   `json:"shape"`
	Datatype string  `json:"datatype"`
	Data     []int64 `json:"data"`
}

type inferRequest struct {
	Inputs  []tensor            `json:"inputs"`
	Outputs []map[string]string `json:"outputs"`
}

type inferResponse struct {
	Outputs []struct {
		Name  string    `json:"name"`
		Shape []int     `json:"shape"`
		Data  []float32 `json:"data"`
	} `json:"outputs"`
	Error string `json:"error"`
}

// Embed returns one L2-normalised 384-d vector per text (normalisation is in the graph).
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += c.MaxBatch {
		vecs, err := c.embedBatch(ctx, texts[start:min(start+c.MaxBatch, len(texts))])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (c *Client) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	encoded := make([][]int64, len(texts))
	longest := 0
	for i, t := range texts {
		encoded[i] = c.tokenizer.Encode(t, MaxTokens)
		longest = max(longest, len(encoded[i]))
	}
	length := BucketLength(longest)
	ids := make([]int64, 0, len(texts)*length)
	mask := make([]int64, 0, len(texts)*length)
	for _, e := range encoded {
		for j := range length {
			if j < len(e) {
				ids, mask = append(ids, e[j]), append(mask, 1)
			} else {
				ids, mask = append(ids, c.tokenizer.PadID()), append(mask, 0)
			}
		}
	}
	shape := []int{len(texts), length}
	body, err := json.Marshal(inferRequest{
		Inputs: []tensor{
			{"input_ids", shape, "INT64", ids},
			{"attention_mask", shape, "INT64", mask},
		},
		Outputs: []map[string]string{{"name": "sentence_embedding"}},
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("%s/v2/models/%s/infer", c.BaseURL, c.Model), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, &Error{Retryable: true, Err: err}
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &Error{Status: resp.StatusCode, Retryable: true, Err: err}
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &Error{
			Status:    resp.StatusCode,
			Retryable: resp.StatusCode >= 500 || resp.StatusCode == http.StatusTooManyRequests,
			Err:       fmt.Errorf("%s", bytes.TrimSpace(raw)),
		}
	}
	var r inferResponse
	if err := json.Unmarshal(raw, &r); err != nil {
		return nil, &Error{Status: resp.StatusCode, Err: err}
	}
	if len(r.Outputs) != 1 || len(r.Outputs[0].Data) != len(texts)*Dim {
		return nil, &Error{Status: resp.StatusCode, Err: fmt.Errorf("unexpected output shape %v", r.Outputs)}
	}
	data := r.Outputs[0].Data
	vecs := make([][]float32, len(texts))
	for i := range vecs {
		vecs[i] = data[i*Dim : (i+1)*Dim]
	}
	return vecs, nil
}
