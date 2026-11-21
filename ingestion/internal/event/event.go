// Package event defines the Kafka records the ingestion pipeline exchanges.
//
//	omni.documents          DocumentEvent, key = "<tenant>/<doc_id>"
//	omni.documents.indexed  IndexedEvent, one per document once it is searchable
//	omni.documents.dlq      the original record plus error headers
package event

import (
	"errors"
	"fmt"
	"regexp"
	"unicode/utf8"
)

const (
	TopicDocuments = "omni.documents"
	TopicIndexed   = "omni.documents.indexed"
	TopicDLQ       = "omni.documents.dlq"

	MaxTextBytes = 1 << 20 // 1 MiB of text per document
)

var idRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$`)

// DocumentEvent is one document to (re)index. Re-sending the same doc_id
// overwrites its chunks: chunk IDs are derived from doc_id and position.
type DocumentEvent struct {
	Tenant     string            `json:"tenant"`
	DocID      string            `json:"doc_id"`
	Title      string            `json:"title"`
	Text       string            `json:"text"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	AcceptedAt int64             `json:"accepted_at_us"` // set by omni-ingest (Unix µs)
}

// Key routes every version of a document to the same partition, so a
// document's updates are applied in order by exactly one worker at a time.
func (d *DocumentEvent) Key() []byte { return []byte(d.Tenant + "/" + d.DocID) }

// Validate rejects events that can never be indexed (they go to the DLQ).
func (d *DocumentEvent) Validate() error {
	var errs []error
	if !idRE.MatchString(d.Tenant) {
		errs = append(errs, fmt.Errorf("invalid tenant %q", d.Tenant))
	}
	if !idRE.MatchString(d.DocID) {
		errs = append(errs, fmt.Errorf("invalid doc_id %q", d.DocID))
	}
	if len(d.Text) == 0 {
		errs = append(errs, errors.New("empty text"))
	}
	if len(d.Text) > MaxTextBytes {
		errs = append(errs, fmt.Errorf("text is %d bytes (max %d)", len(d.Text), MaxTextBytes))
	}
	if !utf8.ValidString(d.Text) || !utf8.ValidString(d.Title) {
		errs = append(errs, errors.New("text and title must be valid UTF-8"))
	}
	return errors.Join(errs...)
}

// IndexedEvent reports that a document's chunks are durable and searchable in
// both Elasticsearch and Qdrant. Delivery is at-least-once: consumers
// deduplicate on (doc_id, partition, offset).
type IndexedEvent struct {
	Tenant     string `json:"tenant"`
	DocID      string `json:"doc_id"`
	Chunks     int    `json:"chunks"`
	Partition  int32  `json:"partition"`
	Offset     int64  `json:"offset"`
	AcceptedAt int64  `json:"accepted_at_us"`
	IndexedAt  int64  `json:"indexed_at_us"`
	Worker     string `json:"worker"`
}
