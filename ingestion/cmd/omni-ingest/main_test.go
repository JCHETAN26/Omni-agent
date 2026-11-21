package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/JCHETAN26/Omni-agent/ingestion/internal/event"
)

func testServer(fail error) (*server, *[]*kgo.Record) {
	var sent []*kgo.Record
	return &server{
		log: slog.New(slog.NewTextHandler(io.Discard, nil)), maxBytes: 1 << 20, maxDocs: 10,
		timeout: time.Second, now: func() time.Time { return time.UnixMicro(42) },
		produceAll: func(_ context.Context, recs ...*kgo.Record) kgo.ProduceResults {
			var res kgo.ProduceResults
			for i, r := range recs {
				r.Partition = int32(i % 2)
				sent = append(sent, r)
				res = append(res, kgo.ProduceResult{Record: r, Err: fail})
			}
			return res
		},
	}, &sent
}

func post(s *server, contentType, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/documents", strings.NewReader(body))
	req.Header.Set("Content-Type", contentType)
	req.Header.Set("X-Tenant", "acme")
	w := httptest.NewRecorder()
	s.handleUpload(w, req)
	return w
}

func TestAcceptedOnlyAfterKafkaAcks(t *testing.T) {
	s, sent := testServer(nil)
	w := post(s, "application/json", `{"documents":[{"doc_id":"a","text":"x"},{"doc_id":"b","text":"y"}]}`)
	if w.Code != http.StatusAccepted {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
	if len(*sent) != 2 || string((*sent)[0].Key) != "acme/a" || (*sent)[0].Topic != event.TopicDocuments {
		t.Fatalf("records: %+v", *sent)
	}
	var d event.DocumentEvent
	_ = json.Unmarshal((*sent)[1].Value, &d)
	if d.Tenant != "acme" || d.AcceptedAt != 42 {
		t.Fatalf("event %+v", d)
	}
}

func TestNDJSONStream(t *testing.T) {
	s, sent := testServer(nil)
	w := post(s, "application/x-ndjson", "{\"doc_id\":\"a\",\"text\":\"x\"}\n\n{\"doc_id\":\"b\",\"text\":\"y\"}\n")
	if w.Code != http.StatusAccepted || len(*sent) != 2 {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
}

func TestInvalidUploadPublishesNothing(t *testing.T) {
	s, sent := testServer(nil)
	w := post(s, "application/json", `{"documents":[{"doc_id":"a","text":"x"},{"doc_id":"bad id!","text":""}]}`)
	if w.Code != http.StatusBadRequest || len(*sent) != 0 {
		t.Fatalf("%d sent=%d", w.Code, len(*sent))
	}
	if post(s, "application/json", `{"documents":[], "extra":1}`).Code != http.StatusBadRequest {
		t.Fatal("unknown fields must be rejected")
	}
}

func TestBrokerFailureIsRetryable(t *testing.T) {
	s, _ := testServer(errors.New("broker down"))
	w := post(s, "application/json", `{"documents":[{"doc_id":"a","text":"x"}]}`)
	if w.Code != http.StatusServiceUnavailable || w.Header().Get("Retry-After") == "" {
		t.Fatalf("%d %v", w.Code, w.Header())
	}
}
