package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	pb "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/upstream"
)

// fakeGateway mimics the FastAPI gateway: /readyz, POST /search, GET /sse/agent
// and POST /vision/classify, each switchable to a failure mode.
type fakeGateway struct {
	*httptest.Server
	name   string
	mode   atomic.Value // "ok", "fail", "hang", "bad", "agent_error"
	calls  atomic.Int64
	lastID atomic.Value
}

var answer = []string{"Replace", " the", " drive", " and", " rebuild."}

// breakNextChat makes exactly one Chat call, on whichever gateway gets it, die
// after two tokens.
var breakNextChat atomic.Bool

func newGateway(name string) *fakeGateway {
	g := &fakeGateway{name: name}
	g.mode.Store("ok")
	g.lastID.Store("")
	g.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/readyz" {
			if g.mode.Load() == "fail" {
				w.WriteHeader(http.StatusServiceUnavailable)
			}
			return
		}
		g.calls.Add(1)
		g.lastID.Store(r.Header.Get("X-Request-ID"))
		switch g.mode.Load() {
		case "fail":
			w.WriteHeader(http.StatusInternalServerError) // e.g. Triton down: hybrid search raises
			return
		case "hang":
			select {
			case <-r.Context().Done():
			case <-time.After(time.Second):
			}
			return
		case "bad":
			w.WriteHeader(http.StatusUnprocessableEntity)
			return
		}
		switch r.URL.Path {
		case "/search":
			var req struct{ Query string }
			_ = json.NewDecoder(r.Body).Decode(&req)
			json.NewEncoder(w).Encode(map[string]any{"hits": []any{map[string]any{
				"chunk": map[string]any{"chunk_id": "c1", "doc_id": name, "text": req.Query}, "score": 0.5}}})
		case "/vision/classify":
			json.NewEncoder(w).Encode(map[string]any{"labels": []any{map[string]any{"label": "kite", "score": 6.2}}})
		case "/sse/agent":
			w.Header().Set("Content-Type", "text/event-stream")
			fl := w.(http.Flusher)
			send := func(ev map[string]any) {
				b, _ := json.Marshal(ev)
				fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev["type"], b)
				fl.Flush()
			}
			send(map[string]any{"type": "agent.step", "node": "route"})
			if g.mode.Load() == "agent_error" {
				send(map[string]any{"type": "response.error", "error": "retrieval failed: triton unavailable"})
				return
			}
			send(map[string]any{"type": "response.sources", "sources": []any{map[string]any{
				"chunk_id": "c1", "doc_id": name, "document_title": "Runbook", "rrf_score": 0.03}}})
			for i, tok := range answer {
				if i == 2 && breakNextChat.CompareAndSwap(true, false) {
					// Die mid-answer: drop the connection without response.end.
					conn, _, _ := w.(http.Hijacker).Hijack()
					conn.Close()
					return
				}
				send(map[string]any{"type": "response.token", "index": i, "text": tok})
			}
			sum := sha256.Sum256([]byte(strings.Join(answer, "")))
			send(map[string]any{"type": "response.end", "outcome": "answered", "tokens": len(answer),
				"sha256": hex.EncodeToString(sum[:])})
		}
	}))
	return g
}

func (g *fakeGateway) addr() string { return strings.TrimPrefix(g.URL, "http://") }

// fakeES answers the BM25 fallback query (or fails, if down is set).
func fakeES(down *atomic.Bool) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if down != nil && down.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"hits": map[string]any{"hits": []any{map[string]any{
			"_score": 7.5, "_source": map[string]any{"chunk_id": "bm25-1", "doc_id": "es", "text": "from bm25"}}}}})
	}))
}

// harness runs a Router server over real gRPC on localhost.
type harness struct {
	srv    *Server
	pool   *upstream.Pool
	router pb.RouterClient
}

func newHarness(t *testing.T, es *httptest.Server, gws ...*fakeGateway) *harness {
	t.Helper()
	var addrs []string
	for _, g := range gws {
		addrs = append(addrs, g.addr())
	}
	pool := upstream.NewPool(upstream.StaticResolver(addrs...), upstream.Config{FailureThreshold: 1})
	_ = pool.Refresh(context.Background())
	pool.ProbeAll(context.Background())
	cfg := Config{Node: &pb.NodeInfo{Id: "n1"}, PerTryTimeout: 300 * time.Millisecond}
	if es != nil {
		cfg.BM25 = NewBM25(es.URL, "omniagent_chunks", "")
	}
	srv := New(pool, cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	g := grpc.NewServer()
	pb.RegisterRouterServer(g, srv)
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go g.Serve(lis)
	t.Cleanup(g.Stop)
	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return &harness{srv, pool, pb.NewRouterClient(conn)}
}

func ctx(t *testing.T) context.Context {
	c, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	return c
}

func TestSearchFailsOverAndEjectsTheFailingGateway(t *testing.T) {
	bad, good := newGateway("bad"), newGateway("good")
	defer bad.Close()
	defer good.Close()
	h := newHarness(t, nil, bad, good)
	bad.mode.Store("fail") // fails after it was admitted as healthy
	for range 10 {
		r, err := h.router.Search(ctx(t), &pb.SearchRequest{Query: "disk error"})
		if err != nil || r.Hits[0].DocId != "good" || r.Route.Degradation != pb.Degradation_DEGRADATION_NONE {
			t.Fatalf("resp=%v err=%v", r, err)
		}
	}
	if n := bad.calls.Load(); n != 1 {
		t.Fatalf("failing gateway called %d times; it must be ejected after its first failure", n)
	}
}

func TestRequestIDIsPropagated(t *testing.T) {
	g := newGateway("g")
	defer g.Close()
	h := newHarness(t, nil, g)
	r, err := h.router.Search(ctx(t), &pb.SearchRequest{Meta: &pb.RequestMeta{RequestId: "req-42"}, Query: "q"})
	if err != nil || r.Route.RequestId != "req-42" || g.lastID.Load() != "req-42" {
		t.Fatalf("route=%v gateway saw %q err=%v", r.GetRoute(), g.lastID.Load(), err)
	}
	r, _ = h.router.Search(ctx(t), &pb.SearchRequest{Query: "q"})
	if r.Route.RequestId == "" || g.lastID.Load() != r.Route.RequestId {
		t.Fatal("a request ID must be generated and forwarded when the caller sends none")
	}
}

func TestDegradesToBM25ThenStaleCacheThenFailsFast(t *testing.T) {
	var esDown atomic.Bool
	es := fakeES(&esDown)
	defer es.Close()
	g := newGateway("g")
	defer g.Close()
	h := newHarness(t, es, g)
	if _, err := h.router.Search(ctx(t), &pb.SearchRequest{Query: "cached"}); err != nil {
		t.Fatal(err)
	}
	g.mode.Store("fail") // Triton or Qdrant down: the gateway's hybrid search returns 500

	r, err := h.router.Search(ctx(t), &pb.SearchRequest{Query: "cached"})
	if err != nil || r.Route.Degradation != pb.Degradation_DEGRADATION_BM25_ONLY || r.Hits[0].ChunkId != "bm25-1" {
		t.Fatalf("want BM25 straight from Elasticsearch, got %v %v", r, err)
	}
	if !strings.Contains(r.Route.DegradedReason, "HTTP 500") {
		t.Fatalf("reason %q", r.Route.DegradedReason)
	}
	esDown.Store(true)
	r, err = h.router.Search(ctx(t), &pb.SearchRequest{Query: "cached"})
	if err != nil || r.Route.Degradation != pb.Degradation_DEGRADATION_STALE_CACHE || r.Hits[0].DocId != "g" {
		t.Fatalf("want the stale cached answer, got %v %v", r, err)
	}
	start := time.Now()
	_, err = h.router.Search(ctx(t), &pb.SearchRequest{Query: "never seen"})
	if status.Code(err) != codes.Unavailable || time.Since(start) > 200*time.Millisecond {
		t.Fatalf("want a fast UNAVAILABLE, got %v after %v", err, time.Since(start))
	}
}

func TestCallerDeadlineBoundsGatewayAttempts(t *testing.T) {
	g := newGateway("slow")
	defer g.Close()
	h := newHarness(t, nil, g)
	g.mode.Store("hang")
	c, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := h.router.Search(c, &pb.SearchRequest{Query: "q"})
	if status.Code(err) != codes.DeadlineExceeded || time.Since(start) > 300*time.Millisecond {
		t.Fatalf("got %v after %v", err, time.Since(start))
	}
	if h.pool.Healthy() != 1 {
		t.Fatal("gateway ejected because the caller's deadline passed")
	}
}

func TestClientErrorsAreNotRetried(t *testing.T) {
	a, b := newGateway("a"), newGateway("b")
	defer a.Close()
	defer b.Close()
	a.mode.Store("bad")
	b.mode.Store("bad")
	h := newHarness(t, nil, a, b)
	_, err := h.router.Search(ctx(t), &pb.SearchRequest{Query: "q"})
	if status.Code(err) != codes.InvalidArgument || a.calls.Load()+b.calls.Load() != 1 {
		t.Fatalf("err=%v calls=%d", err, a.calls.Load()+b.calls.Load())
	}
	if _, err := h.router.Search(ctx(t), &pb.SearchRequest{Query: "q", TopK: 500}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("top_k validation: %v", err)
	}
}

func collect(t *testing.T, h *harness, q string) ([]*pb.ChatEvent, error) {
	t.Helper()
	stream, err := h.router.Chat(ctx(t), &pb.ChatRequest{Question: q})
	if err != nil {
		return nil, err
	}
	var evs []*pb.ChatEvent
	for {
		ev, err := stream.Recv()
		if err == io.EOF {
			return evs, nil
		}
		if err != nil {
			return evs, err
		}
		evs = append(evs, ev)
	}
}

// tokensSinceRestart rebuilds the answer the way a client must: drop
// everything received before the last Restart.
func tokensSinceRestart(evs []*pb.ChatEvent) (string, int) {
	var b strings.Builder
	restarts := 0
	for _, ev := range evs {
		switch {
		case ev.GetRestart() != nil:
			b.Reset()
			restarts++
		case ev.GetToken() != nil:
			b.WriteString(ev.GetToken().Text)
		}
	}
	return b.String(), restarts
}

func TestChatStreamsTheAgentTurn(t *testing.T) {
	g := newGateway("g")
	defer g.Close()
	h := newHarness(t, nil, g)
	evs, err := collect(t, h, "how do I fix EIO?")
	if err != nil {
		t.Fatal(err)
	}
	text, restarts := tokensSinceRestart(evs)
	done := evs[len(evs)-1]
	sum := sha256.Sum256([]byte(text))
	if text != strings.Join(answer, "") || restarts != 0 || done.GetDone().Sha256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("text %q restarts %d done %v", text, restarts, done)
	}
	if evs[0].GetSources().Hits[0].DocumentTitle != "Runbook" || done.Route.Upstream == "" {
		t.Fatalf("sources %v route %v", evs[0], done.Route)
	}
}

func TestChatRestartsOnAnotherGatewayWhenOneDiesMidAnswer(t *testing.T) {
	a, b := newGateway("a"), newGateway("b")
	defer a.Close()
	defer b.Close()
	h := newHarness(t, nil, a, b)
	breakNextChat.Store(true) // the first gateway picked dies after two tokens
	evs, err := collect(t, h, "q")
	if err != nil {
		t.Fatal(err)
	}
	text, restarts := tokensSinceRestart(evs)
	done := evs[len(evs)-1]
	if text != strings.Join(answer, "") || restarts != 1 || done.Route.Attempts != 2 {
		t.Fatalf("text %q restarts %d route %v", text, restarts, done.Route)
	}
}

func TestChatDegradesToPassagesWhenNoAgentCanAnswer(t *testing.T) {
	es := fakeES(nil)
	defer es.Close()
	g := newGateway("g")
	defer g.Close()
	h := newHarness(t, es, g)
	g.mode.Store("agent_error")
	evs, err := collect(t, h, "q")
	if err != nil {
		t.Fatal(err)
	}
	done := evs[len(evs)-1]
	if done.Route.Degradation != pb.Degradation_DEGRADATION_PASSAGES_ONLY || done.GetDone().Outcome != "degraded" {
		t.Fatalf("done %v", done)
	}
	if evs[0].GetSources().Hits[0].ChunkId != "bm25-1" || !strings.Contains(done.Route.DegradedReason, "triton") {
		t.Fatalf("events %v", evs)
	}
}

func TestVisionFailsOverAndNeverInventsLabels(t *testing.T) {
	a, b := newGateway("a"), newGateway("b")
	defer a.Close()
	defer b.Close()
	h := newHarness(t, fakeES(nil), a, b)
	a.mode.Store("fail")
	r, err := h.router.Vision(ctx(t), &pb.VisionRequest{Image: []byte("jpeg"), ThreadId: "t1"})
	if err != nil || r.Labels[0].Label != "kite" || r.Route.Upstream != b.addr() {
		t.Fatalf("resp %v err %v", r, err)
	}
	b.mode.Store("fail")
	if _, err := h.router.Vision(ctx(t), &pb.VisionRequest{Image: []byte("jpeg")}); status.Code(err) != codes.Unavailable {
		t.Fatalf("want UNAVAILABLE with no working gateway, got %v", err)
	}
}

func TestStatusAndDrain(t *testing.T) {
	g := newGateway("g")
	defer g.Close()
	h := newHarness(t, nil, g)
	_, _ = h.router.Search(ctx(t), &pb.SearchRequest{Query: "q"})
	if st := h.srv.Status(); st.RequestsTotal != 1 || st.State != pb.NodeState_NODE_STATE_ALIVE {
		t.Fatalf("%v", st)
	}
	h.srv.Drain()
	if h.srv.Status().State != pb.NodeState_NODE_STATE_DRAINING {
		t.Fatal("drain not reported")
	}
}
