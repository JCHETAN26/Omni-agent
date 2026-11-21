package embed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRequestShapeBucketAndSplit(t *testing.T) {
	var shapes [][]int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/models/minilm_embedder/infer" {
			t.Errorf("path %s", r.URL.Path)
		}
		var req inferRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		shapes = append(shapes, req.Inputs[0].Shape)
		ids, mask := req.Inputs[0].Data, req.Inputs[1].Data
		if ids[0] != 101 || mask[0] != 1 || mask[len(mask)-1] != 0 {
			t.Errorf("ids[0]=%d mask=%v", ids[0], mask[:4])
		}
		n := req.Inputs[0].Shape[0]
		data := make([]float32, n*Dim)
		json.NewEncoder(w).Encode(map[string]any{"outputs": []any{map[string]any{
			"name": "sentence_embedding", "shape": []int{n, Dim}, "data": data}}})
	}))
	defer srv.Close()
	c, err := New(srv.URL, "minilm_embedder", 2)
	if err != nil {
		t.Fatal(err)
	}
	vecs, err := c.Embed(context.Background(), []string{"hello world", "a", "kafka broker"})
	if err != nil || len(vecs) != 3 || len(vecs[2]) != Dim {
		t.Fatalf("%v %d", err, len(vecs))
	}
	if len(shapes) != 2 || shapes[0][0] != 2 || shapes[0][1] != 32 || shapes[1][0] != 1 {
		t.Fatalf("shapes %v", shapes)
	}
}

func TestBucketLength(t *testing.T) {
	for n, want := range map[int]int{1: 32, 32: 32, 33: 64, 200: 256, 999: 256} {
		if got := BucketLength(n); got != want {
			t.Errorf("BucketLength(%d)=%d want %d", n, got, want)
		}
	}
}
