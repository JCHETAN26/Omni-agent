// Package client talks to an OmniAgent gateway: readiness, ingestion and
// streamed agent answers over SSE, with event-sequence and checksum checks.
package client

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// GatewayError means the gateway answered with an error or an incomplete stream.
type GatewayError struct{ Msg string }

func (e *GatewayError) Error() string { return e.Msg }

func gatewayErr(format string, args ...any) error {
	return &GatewayError{Msg: fmt.Sprintf(format, args...)}
}

type Client struct {
	Server string
	HTTP   *http.Client
}

func New(server string) *Client {
	return &Client{Server: strings.TrimRight(server, "/"), HTTP: &http.Client{Timeout: 90 * time.Second}}
}

// ValidateServer accepts only plain http(s) URLs with a host and no
// credentials, query or fragment.
func ValidateServer(raw string) (string, error) {
	server := strings.TrimRight(raw, "/")
	u, err := url.Parse(server)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Hostname() == "" ||
		u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return "", errors.New("--server must be an HTTP or HTTPS URL")
	}
	return server, nil
}

func (c *Client) getJSON(ctx context.Context, path string, body any, out any) error {
	method, reader := http.MethodGet, io.Reader(nil)
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return err
		}
		method, reader = http.MethodPost, bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.Server+path, reader)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return gatewayErr("%s %s: HTTP %d %s", method, path, resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// Ready reports whether the gateway's /readyz says it can serve.
func (c *Client) Ready(ctx context.Context) (bool, error) {
	var out struct {
		Ready bool `json:"ready"`
	}
	if err := c.getJSON(ctx, "/readyz", nil, &out); err != nil {
		return false, err
	}
	return out.Ready, nil
}

type Document struct {
	DocID string `json:"doc_id"`
	Title string `json:"title"`
	Text  string `json:"text"`
}

// CollectRunbooks reads nonempty Markdown files under dir (no symlinks). Each
// document id is "<dir name>/<path relative to dir>", so re-uploads upsert.
func CollectRunbooks(dir string) ([]Document, error) {
	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		return nil, fmt.Errorf("expected a runbook directory: %s", dir)
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}
	var docs []Document
	err = filepath.WalkDir(abs, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type()&fs.ModeSymlink != 0 || d.IsDir() || !strings.HasSuffix(path, ".md") {
			return nil
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		text := strings.TrimSpace(string(raw))
		if text == "" {
			return nil
		}
		rel, _ := filepath.Rel(abs, path)
		title := strings.TrimSuffix(filepath.Base(path), ".md")
		for _, line := range strings.Split(text, "\n") {
			if strings.HasPrefix(line, "# ") {
				title = strings.TrimSpace(line[2:])
				break
			}
		}
		docs = append(docs, Document{
			DocID: filepath.Base(abs) + "/" + filepath.ToSlash(rel),
			Title: title,
			Text:  text,
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(docs) == 0 {
		return nil, errors.New("no nonempty Markdown runbooks found")
	}
	sort.Slice(docs, func(i, j int) bool { return docs[i].DocID < docs[j].DocID })
	return docs, nil
}

// Ingest uploads documents and returns the gateway's ingestion report.
func (c *Client) Ingest(ctx context.Context, docs []Document) (map[string]any, error) {
	var out map[string]any
	err := c.getJSON(ctx, "/ingest", map[string]any{"documents": docs}, &out)
	return out, err
}

type Source map[string]any

type Answer struct {
	Text     string   `json:"text"`
	Sources  []Source `json:"sources"`
	Outcome  string   `json:"outcome"`
	Tool     string   `json:"tool"`
	Retries  int      `json:"retries"`
	ThreadID string   `json:"thread_id"`
	TTFTms   *float64 `json:"ttft_ms"`
	TotalMs  float64  `json:"total_ms"`
}

type event struct {
	Type    string   `json:"type"`
	Error   string   `json:"error"`
	Sources []Source `json:"sources"`
	Index   int      `json:"index"`
	Text    string   `json:"text"`
	Tokens  int      `json:"tokens"`
	SHA256  string   `json:"sha256"`
	Outcome string   `json:"outcome"`
	Tool    string   `json:"tool"`
	Retries int      `json:"retries"`
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// Ask streams one agent turn from /sse/agent. It rejects out-of-order events,
// out-of-order tokens, and answers whose token count or SHA-256 differ from
// what the gateway reports in response.end.
func (c *Client) Ask(ctx context.Context, question, threadID string) (*Answer, error) {
	params := url.Values{"q": {question}}
	if threadID != "" {
		params.Set("thread_id", threadID)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.Server+"/sse/agent?"+params.Encode(), nil)
	if err != nil {
		return nil, err
	}
	start := time.Now()
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, gatewayErr("/sse/agent: HTTP %d", resp.StatusCode)
	}
	return readStream(resp.Body, resp.Header.Get("X-Thread-Id"), start)
}

func readStream(body io.Reader, thread string, start time.Time) (*Answer, error) {
	var (
		parts   []string
		sources []Source
		first   time.Duration
		eventID int
	)
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 8*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "id: "):
			eventID++
			if n, err := strconv.Atoi(line[4:]); err != nil || n != eventID {
				return nil, gatewayErr("unexpected SSE event sequence")
			}
		case strings.HasPrefix(line, "data: "):
			var ev event
			if err := json.Unmarshal([]byte(line[6:]), &ev); err != nil {
				return nil, gatewayErr("malformed SSE event: %v", err)
			}
			switch ev.Type {
			case "response.error":
				if ev.Error == "" {
					ev.Error = "agent request failed"
				}
				return nil, gatewayErr("%s", ev.Error)
			case "response.sources":
				sources = ev.Sources
			case "response.token":
				if ev.Index != len(parts) {
					return nil, gatewayErr("unexpected answer token sequence")
				}
				if first == 0 {
					first = time.Since(start)
				}
				parts = append(parts, ev.Text)
			case "response.end":
				text := strings.Join(parts, "")
				sum := sha256.Sum256([]byte(text))
				if len(parts) != ev.Tokens || hex.EncodeToString(sum[:]) != ev.SHA256 {
					return nil, gatewayErr("answer integrity check failed")
				}
				ans := &Answer{
					Text: text, Sources: sources, Outcome: ev.Outcome, Tool: ev.Tool,
					Retries: ev.Retries, ThreadID: thread, TotalMs: ms(time.Since(start)),
				}
				if sources == nil {
					ans.Sources = []Source{}
				}
				if first > 0 {
					t := ms(first)
					ans.TTFTms = &t
				}
				return ans, nil
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return nil, gatewayErr("stream ended without a complete answer")
}
