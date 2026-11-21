// Package server implements the Router service (omni.control.v1).
package server

import (
	"bufio"
	"bytes"
	"container/list"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	pb "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/metrics"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/upstream"
)

// Health service names clients check.
const (
	RouterService  = "omni.control.v1.Router"
	ClusterService = "omni.control.v1.Cluster"
)

type Config struct {
	Node           *pb.NodeInfo
	DefaultTimeout time.Duration // when the caller sets no deadline (default 5s; Chat 60s)
	PerTryTimeout  time.Duration // cap per gateway attempt for Search/Vision (default 2s)
	MaxAttempts    int           // gateway attempts per call (default 3)
	CacheSize      int           // last-good Search answers for degraded mode (default 10000)
	CacheTTL       time.Duration // default 10m
	BM25           *BM25         // nil disables the BM25 fallback
}

type Server struct {
	pb.UnimplementedRouterServer
	cfg      Config
	pool     *upstream.Pool
	http     *http.Client
	Health   *health.Server
	log      *slog.Logger
	inflight atomic.Int64
	total    atomic.Int64
	draining atomic.Bool
	rate     *window
	cache    *lru
	samples  *sampler
}

func New(pool *upstream.Pool, cfg Config, log *slog.Logger) *Server {
	if cfg.DefaultTimeout == 0 {
		cfg.DefaultTimeout = 5 * time.Second
	}
	if cfg.PerTryTimeout == 0 {
		cfg.PerTryTimeout = 2 * time.Second
	}
	if cfg.MaxAttempts == 0 {
		cfg.MaxAttempts = 3
	}
	if cfg.CacheSize == 0 {
		cfg.CacheSize = 10000
	}
	if cfg.CacheTTL == 0 {
		cfg.CacheTTL = 10 * time.Minute
	}
	if cfg.Node == nil {
		cfg.Node = &pb.NodeInfo{Id: "node"}
	}
	s := &Server{
		cfg: cfg, pool: pool, log: log, Health: health.NewServer(),
		http:    &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 64, IdleConnTimeout: 30 * time.Second}},
		rate:    newWindow(10),
		cache:   newLRU(cfg.CacheSize, cfg.CacheTTL),
		samples: &sampler{buf: make([]int64, 100)},
	}
	for _, name := range []string{"", RouterService, ClusterService} {
		s.Health.SetServingStatus(name, healthpb.HealthCheckResponse_SERVING)
	}
	return s
}

// Drain reports NOT_SERVING so health-checking clients stop sending new calls
// here; in-flight calls continue. Call before GracefulStop.
func (s *Server) Drain() {
	s.draining.Store(true)
	for _, name := range []string{"", RouterService, ClusterService} {
		s.Health.SetServingStatus(name, healthpb.HealthCheckResponse_NOT_SERVING)
	}
}

// Status is this node's live status (for Cluster heartbeats).
func (s *Server) Status() *pb.NodeStatus {
	st := pb.NodeState_NODE_STATE_ALIVE
	if s.draining.Load() {
		st = pb.NodeState_NODE_STATE_DRAINING
	}
	return &pb.NodeStatus{
		Node: s.cfg.Node, State: st, LastHeartbeatUnixMs: time.Now().UnixMilli(),
		Inflight: s.inflight.Load(), InflightAvg_10S: s.samples.mean(), Rps: s.rate.rate(time.Now()),
		RequestsTotal: s.total.Load(),
	}
}

// --- request bookkeeping --------------------------------------------------

func newRequestID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

func requestID(ctx context.Context, meta *pb.RequestMeta) string {
	if meta != nil && meta.RequestId != "" {
		return meta.RequestId
	}
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if v := md.Get("x-request-id"); len(v) > 0 && v[0] != "" {
			return v[0]
		}
	}
	return newRequestID()
}

// begin counts a call and returns its finisher.
func (s *Server) begin(method string) func(err error, route *pb.RouteInfo) {
	start := time.Now()
	s.inflight.Add(1)
	metrics.Inflight.Inc()
	s.total.Add(1)
	s.rate.add(start)
	return func(err error, route *pb.RouteInfo) {
		s.inflight.Add(-1)
		metrics.Inflight.Dec()
		deg := pb.Degradation_DEGRADATION_NONE
		if route != nil {
			deg = route.Degradation
		}
		metrics.Requests.WithLabelValues(method, status.Code(err).String(), degradationLabel(deg)).Inc()
		metrics.Latency.WithLabelValues(method).Observe(time.Since(start).Seconds())
	}
}

func degradationLabel(d pb.Degradation) string {
	return strings.ToLower(strings.TrimPrefix(d.String(), "DEGRADATION_"))
}

func withDefaultDeadline(ctx context.Context, d time.Duration) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, d)
}

// errClient is a 4xx from the gateway: the request is bad, not the endpoint.
type errClient struct{ msg string }

func (e errClient) Error() string { return e.msg }

// attempt is one gateway call; it reports whether output already reached the client.
type attempt func(ctx context.Context, addr string) error

// route tries healthy gateways (never the same one twice) until one succeeds,
// a client error ends the call, the caller's deadline passes, or MaxAttempts
// is used. It returns the gateway that answered, the attempt count and the
// last error.
func (s *Server) route(ctx context.Context, method string, call attempt) (string, int, error) {
	tried := map[string]bool{}
	var lastErr error
	attempts := 0
	for attempts < s.cfg.MaxAttempts && ctx.Err() == nil {
		ep := s.pool.Pick(tried)
		if ep == nil {
			if lastErr == nil {
				lastErr = errors.New("no healthy gateway")
			}
			break
		}
		if attempts > 0 {
			metrics.Failovers.WithLabelValues(method).Inc()
		}
		attempts++
		tried[ep.Addr] = true
		done := s.pool.Begin(ep)
		err := call(ctx, ep.Addr)
		var ce errClient
		switch {
		case err == nil:
			done(true)
			metrics.Attempts.WithLabelValues("ok").Inc()
			return ep.Addr, attempts, nil
		case errors.As(err, &ce):
			done(true) // the endpoint answered correctly; the request is invalid
			metrics.Attempts.WithLabelValues("client_error").Inc()
			return ep.Addr, attempts, err
		default:
			done(ctx.Err() != nil) // our own deadline passing is not the endpoint's fault
			metrics.Attempts.WithLabelValues("retryable_error").Inc()
			lastErr = err
		}
	}
	if ctx.Err() != nil && lastErr == nil {
		lastErr = ctx.Err()
	}
	return "", attempts, lastErr
}

func (s *Server) newRequest(ctx context.Context, method, addr, path string, body io.Reader, rid string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, "http://"+addr+path, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Request-ID", rid)
	if dl, ok := ctx.Deadline(); ok {
		req.Header.Set("X-Request-Deadline-Ms", fmt.Sprint(time.Until(dl).Milliseconds()))
	}
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if v := md.Get("traceparent"); len(v) > 0 {
			req.Header.Set("traceparent", v[0])
		}
	}
	return req, nil
}

// classify turns a gateway HTTP status into an error (nil for 200).
func classify(addr string, resp *http.Response, raw []byte) error {
	switch c := resp.StatusCode; {
	case c == http.StatusOK:
		return nil
	case c >= 400 && c < 500 && c != http.StatusTooManyRequests && c != http.StatusRequestTimeout:
		return errClient{fmt.Sprintf("gateway rejected the request (HTTP %d): %.200s", c, raw)}
	default:
		return fmt.Errorf("gateway %s: HTTP %d: %.120s", addr, c, raw)
	}
}

// --- Search ---------------------------------------------------------------

type gatewayHit struct {
	Chunk struct {
		ChunkID        string `json:"chunk_id"`
		DocID          string `json:"doc_id"`
		DocumentTitle  string `json:"document_title"`
		SectionHeading string `json:"section_heading"`
		Text           string `json:"text"`
	} `json:"chunk"`
	Score float64 `json:"score"`
}

func toHits(in []gatewayHit) []*pb.Hit {
	out := make([]*pb.Hit, len(in))
	for i, h := range in {
		out[i] = &pb.Hit{ChunkId: h.Chunk.ChunkID, DocId: h.Chunk.DocID, DocumentTitle: h.Chunk.DocumentTitle,
			SectionHeading: h.Chunk.SectionHeading, Text: h.Chunk.Text, Score: h.Score}
	}
	return out
}

func (s *Server) Search(ctx context.Context, req *pb.SearchRequest) (resp *pb.SearchResponse, err error) {
	finish := s.begin("Search")
	defer func() {
		var r *pb.RouteInfo
		if resp != nil {
			r = resp.Route
		}
		finish(err, r)
	}()
	if req.Query == "" || len(req.Query) > 2000 {
		return nil, status.Error(codes.InvalidArgument, "query must be 1-2000 bytes")
	}
	topK := req.TopK
	if topK == 0 {
		topK = 10
	}
	if topK < 1 || topK > 100 {
		return nil, status.Error(codes.InvalidArgument, "top_k must be 1-100")
	}
	ctx, cancel := withDefaultDeadline(ctx, s.cfg.DefaultTimeout)
	defer cancel()
	rid := requestID(ctx, req.Meta)
	route := &pb.RouteInfo{RequestId: rid, Node: s.cfg.Node.Id}
	key := fmt.Sprintf("%d\x00%s", topK, req.Query)
	body, _ := json.Marshal(map[string]any{"query": req.Query, "top_k": topK})

	var hits []*pb.Hit
	var upstreamMs float64
	addr, attempts, lastErr := s.route(ctx, "Search", func(ctx context.Context, addr string) error {
		tctx, tcancel := context.WithTimeout(ctx, s.cfg.PerTryTimeout)
		defer tcancel()
		t0 := time.Now()
		hreq, err := s.newRequest(tctx, http.MethodPost, addr, "/search", bytes.NewReader(body), rid)
		if err != nil {
			return err
		}
		hreq.Header.Set("Content-Type", "application/json")
		hresp, err := s.http.Do(hreq)
		if err != nil {
			return err
		}
		defer hresp.Body.Close()
		raw, err := io.ReadAll(hresp.Body)
		if err != nil {
			return err
		}
		if err := classify(addr, hresp, raw); err != nil {
			return err
		}
		var out struct {
			Hits []gatewayHit `json:"hits"`
		}
		if err := json.Unmarshal(raw, &out); err != nil {
			return fmt.Errorf("gateway %s: bad response: %w", addr, err)
		}
		hits, upstreamMs = toHits(out.Hits), float64(time.Since(t0).Microseconds())/1000
		return nil
	})
	route.Attempts = int32(attempts)
	if lastErr == nil {
		route.Upstream, route.UpstreamMs = addr, upstreamMs
		s.cache.put(key, hits)
		return &pb.SearchResponse{Hits: hits, Route: route}, nil
	}
	var ce errClient
	if errors.As(lastErr, &ce) {
		return nil, status.Error(codes.InvalidArgument, ce.msg)
	}
	return s.degradeSearch(ctx, req.Query, topK, key, route, lastErr)
}

// degradeSearch: BM25 straight from Elasticsearch, else the last good answer,
// else UNAVAILABLE. Never a hang: every step is bounded by the caller's deadline.
func (s *Server) degradeSearch(ctx context.Context, query string, topK int32, key string,
	route *pb.RouteInfo, cause error) (*pb.SearchResponse, error) {
	route.DegradedReason = cause.Error()
	if s.cfg.BM25 != nil && ctx.Err() == nil {
		if hits, err := s.cfg.BM25.Search(ctx, query, topK); err == nil {
			route.Degradation = pb.Degradation_DEGRADATION_BM25_ONLY
			metrics.Degraded.WithLabelValues("bm25_only").Inc()
			return &pb.SearchResponse{Hits: hits, Route: route}, nil
		} else {
			route.DegradedReason += "; bm25: " + err.Error()
		}
	}
	if hits, ok := s.cache.get(key); ok {
		route.Degradation = pb.Degradation_DEGRADATION_STALE_CACHE
		metrics.Degraded.WithLabelValues("stale_cache").Inc()
		return &pb.SearchResponse{Hits: hits, Route: route}, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, status.FromContextError(err).Err()
	}
	return nil, status.Errorf(codes.Unavailable, "search unavailable: %s", route.DegradedReason)
}

// --- Chat -----------------------------------------------------------------

const degradedChatNotice = "The assistant is unavailable right now. These are the most relevant passages from the knowledge base:"

func (s *Server) Chat(req *pb.ChatRequest, stream grpc.ServerStreamingServer[pb.ChatEvent]) (err error) {
	finish := s.begin("Chat")
	route := &pb.RouteInfo{Node: s.cfg.Node.Id}
	defer func() { finish(err, route) }()
	if req.Question == "" || len(req.Question) > 4000 {
		return status.Error(codes.InvalidArgument, "question must be 1-4000 bytes")
	}
	ctx, cancel := withDefaultDeadline(stream.Context(), 60*time.Second)
	defer cancel()
	route.RequestId = requestID(ctx, req.Meta)
	q := url.Values{"q": {req.Question}}
	if req.ThreadId != "" {
		q.Set("thread_id", req.ThreadId)
	}
	sent := false // output already reached the client in this call
	var done *pb.Done
	addr, attempts, lastErr := s.route(ctx, "Chat", func(ctx context.Context, addr string) error {
		if sent {
			metrics.ChatRestarts.Inc()
			if err := stream.Send(&pb.ChatEvent{Event: &pb.ChatEvent_Restart{Restart: &pb.Restart{
				Reason: "upstream gateway failed mid-answer; restarting on another gateway"}}}); err != nil {
				return errClient{err.Error()} // the client went away: stop
			}
			sent = false
		}
		var err error
		done, err = s.relayChat(ctx, addr, "/sse/agent?"+q.Encode(), route.RequestId, stream, &sent)
		return err
	})
	route.Attempts = int32(attempts)
	if lastErr == nil {
		route.Upstream = addr
		return stream.Send(&pb.ChatEvent{Event: &pb.ChatEvent_Done{Done: done}, Route: route})
	}
	var ce errClient
	if errors.As(lastErr, &ce) {
		return status.Error(codes.InvalidArgument, ce.msg)
	}
	// Degraded: no gateway could run the agent. Offer the best BM25 passages.
	route.DegradedReason = lastErr.Error()
	if s.cfg.BM25 == nil || ctx.Err() != nil {
		return status.Errorf(codes.Unavailable, "chat unavailable: %v", lastErr)
	}
	hits, err := s.cfg.BM25.Search(ctx, req.Question, 5)
	if err != nil {
		return status.Errorf(codes.Unavailable, "chat unavailable: %v; bm25: %v", lastErr, err)
	}
	if sent {
		_ = stream.Send(&pb.ChatEvent{Event: &pb.ChatEvent_Restart{Restart: &pb.Restart{Reason: "degraded"}}})
	}
	route.Degradation = pb.Degradation_DEGRADATION_PASSAGES_ONLY
	metrics.Degraded.WithLabelValues("passages_only").Inc()
	if err := stream.Send(&pb.ChatEvent{Event: &pb.ChatEvent_Sources{Sources: &pb.Sources{Hits: hits}}}); err != nil {
		return err
	}
	if err := stream.Send(&pb.ChatEvent{Event: &pb.ChatEvent_Token{Token: &pb.Token{Index: 0, Text: degradedChatNotice}}}); err != nil {
		return err
	}
	sum := sha256.Sum256([]byte(degradedChatNotice))
	return stream.Send(&pb.ChatEvent{Event: &pb.ChatEvent_Done{Done: &pb.Done{
		Outcome: "degraded", Tokens: 1, Sha256: hex.EncodeToString(sum[:])}}, Route: route})
}

// agentSource is one citation in the agent's response.sources event.
type agentSource struct {
	ChunkID        string  `json:"chunk_id"`
	DocID          string  `json:"doc_id"`
	DocumentTitle  string  `json:"document_title"`
	SectionHeading string  `json:"section_heading"`
	RRFScore       float64 `json:"rrf_score"`
}

// relayChat streams one /sse/agent turn from addr to the client.
func (s *Server) relayChat(ctx context.Context, addr, path, rid string,
	stream grpc.ServerStreamingServer[pb.ChatEvent], sent *bool) (*pb.Done, error) {
	hreq, err := s.newRequest(ctx, http.MethodGet, addr, path, nil, rid)
	if err != nil {
		return nil, err
	}
	hresp, err := s.http.Do(hreq)
	if err != nil {
		return nil, err
	}
	defer hresp.Body.Close()
	if hresp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(hresp.Body, 512))
		return nil, classify(addr, hresp, raw)
	}
	sc := bufio.NewScanner(hresp.Body)
	sc.Buffer(make([]byte, 64*1024), 4<<20)
	for sc.Scan() {
		line := sc.Bytes()
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		var ev struct {
			Type    string        `json:"type"`
			Index   int32         `json:"index"`
			Text    string        `json:"text"`
			Sources []agentSource `json:"sources"`
			Outcome string        `json:"outcome"`
			Tokens  int32         `json:"tokens"`
			Sha256  string        `json:"sha256"`
			Error   string        `json:"error"`
		}
		if err := json.Unmarshal(line[6:], &ev); err != nil {
			continue
		}
		var out *pb.ChatEvent
		switch ev.Type {
		case "response.sources":
			hits := make([]*pb.Hit, len(ev.Sources))
			for i, src := range ev.Sources {
				hits[i] = &pb.Hit{ChunkId: src.ChunkID, DocId: src.DocID, DocumentTitle: src.DocumentTitle,
					SectionHeading: src.SectionHeading, Score: src.RRFScore}
			}
			out = &pb.ChatEvent{Event: &pb.ChatEvent_Sources{Sources: &pb.Sources{Hits: hits}}}
		case "response.token":
			out = &pb.ChatEvent{Event: &pb.ChatEvent_Token{Token: &pb.Token{Index: ev.Index, Text: ev.Text}}}
		case "response.end":
			return &pb.Done{Outcome: ev.Outcome, Tokens: ev.Tokens, Sha256: ev.Sha256}, nil
		case "response.error":
			// The gateway is up but the agent failed (e.g. a dependency is down).
			return nil, fmt.Errorf("gateway %s: agent error: %s", addr, ev.Error)
		default:
			continue // steps, warnings: not part of the RPC contract
		}
		if err := stream.Send(out); err != nil {
			return nil, errClient{err.Error()} // the client went away: stop
		}
		*sent = true
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("gateway %s: stream broke mid-answer: %w", addr, err)
	}
	return nil, fmt.Errorf("gateway %s: stream ended without response.end", addr)
}

// --- Vision ---------------------------------------------------------------

func (s *Server) Vision(ctx context.Context, req *pb.VisionRequest) (resp *pb.VisionResponse, err error) {
	finish := s.begin("Vision")
	defer func() {
		var r *pb.RouteInfo
		if resp != nil {
			r = resp.Route
		}
		finish(err, r)
	}()
	if len(req.Image) == 0 || len(req.Image) > 4<<20 {
		return nil, status.Error(codes.InvalidArgument, "image must be 1 byte to 4 MiB")
	}
	ctx, cancel := withDefaultDeadline(ctx, s.cfg.DefaultTimeout)
	defer cancel()
	rid := requestID(ctx, req.Meta)
	path := "/vision/classify"
	if req.ThreadId != "" {
		path += "?" + url.Values{"thread_id": {req.ThreadId}}.Encode()
	}
	var labels []*pb.Label
	var upstreamMs float64
	addr, attempts, lastErr := s.route(ctx, "Vision", func(ctx context.Context, addr string) error {
		tctx, tcancel := context.WithTimeout(ctx, s.cfg.PerTryTimeout)
		defer tcancel()
		t0 := time.Now()
		hreq, err := s.newRequest(tctx, http.MethodPost, addr, path, bytes.NewReader(req.Image), rid)
		if err != nil {
			return err
		}
		hreq.Header.Set("Content-Type", "application/octet-stream")
		hresp, err := s.http.Do(hreq)
		if err != nil {
			return err
		}
		defer hresp.Body.Close()
		raw, err := io.ReadAll(hresp.Body)
		if err != nil {
			return err
		}
		if err := classify(addr, hresp, raw); err != nil {
			return err
		}
		var out struct {
			Labels []struct {
				Label string  `json:"label"`
				Score float64 `json:"score"`
			} `json:"labels"`
		}
		if err := json.Unmarshal(raw, &out); err != nil {
			return fmt.Errorf("gateway %s: bad response: %w", addr, err)
		}
		for _, l := range out.Labels {
			labels = append(labels, &pb.Label{Label: l.Label, Score: l.Score})
		}
		upstreamMs = float64(time.Since(t0).Microseconds()) / 1000
		return nil
	})
	route := &pb.RouteInfo{RequestId: rid, Node: s.cfg.Node.Id, Attempts: int32(attempts)}
	if lastErr == nil {
		route.Upstream, route.UpstreamMs = addr, upstreamMs
		return &pb.VisionResponse{Labels: labels, Route: route}, nil
	}
	var ce errClient
	if errors.As(lastErr, &ce) {
		return nil, status.Error(codes.InvalidArgument, ce.msg)
	}
	if err := ctx.Err(); err != nil {
		return nil, status.FromContextError(err).Err()
	}
	// No degraded labels: guessing what a camera shows would mislead the agent.
	return nil, status.Errorf(codes.Unavailable, "vision unavailable: %v", lastErr)
}

// --- load sampling, rate window, cache --------------------------------------

// SampleInflight records the in-flight count every 100 ms until ctx ends.
func (s *Server) SampleInflight(ctx context.Context) {
	t := time.NewTicker(100 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.samples.add(s.inflight.Load())
		}
	}
}

type sampler struct {
	mu   sync.Mutex
	buf  []int64
	n, i int
}

func (m *sampler) add(v int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.buf[m.i] = v
	m.i = (m.i + 1) % len(m.buf)
	m.n = min(m.n+1, len(m.buf))
}

func (m *sampler) mean() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.n == 0 {
		return 0
	}
	var sum int64
	for _, v := range m.buf[:m.n] {
		sum += v
	}
	return float64(sum) / float64(m.n)
}

// window counts events per second over the last n seconds.
type window struct {
	mu      sync.Mutex
	n       int64
	buckets []int64
	stamps  []int64
}

func newWindow(n int) *window {
	return &window{n: int64(n), buckets: make([]int64, n), stamps: make([]int64, n)}
}

func (w *window) add(t time.Time) {
	sec := t.Unix()
	i := sec % w.n
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.stamps[i] != sec {
		w.stamps[i], w.buckets[i] = sec, 0
	}
	w.buckets[i]++
}

func (w *window) rate(t time.Time) float64 {
	now := t.Unix()
	w.mu.Lock()
	defer w.mu.Unlock()
	var sum int64
	for i := range w.buckets {
		if age := now - w.stamps[i]; age >= 1 && age <= w.n {
			sum += w.buckets[i]
		}
	}
	return float64(sum) / float64(w.n)
}

// lru keeps the last good Search answer per request for degraded mode.
type lru struct {
	mu    sync.Mutex
	cap   int
	ttl   time.Duration
	order *list.List
	items map[string]*list.Element
}

type entry struct {
	key  string
	hits []*pb.Hit
	at   time.Time
}

func newLRU(capacity int, ttl time.Duration) *lru {
	return &lru{cap: capacity, ttl: ttl, order: list.New(), items: map[string]*list.Element{}}
}

func (c *lru) put(key string, hits []*pb.Hit) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		el.Value = &entry{key, hits, time.Now()}
		c.order.MoveToFront(el)
		return
	}
	c.items[key] = c.order.PushFront(&entry{key, hits, time.Now()})
	if c.order.Len() > c.cap {
		old := c.order.Back()
		c.order.Remove(old)
		delete(c.items, old.Value.(*entry).key)
	}
}

func (c *lru) get(key string) ([]*pb.Hit, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.items[key]
	if !ok || time.Since(el.Value.(*entry).at) > c.ttl {
		return nil, false
	}
	return el.Value.(*entry).hits, true
}
