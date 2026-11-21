// Package upstream tracks the gateway endpoints a control-plane node routes to.
//
// Each endpoint is in one of three states:
//
//	HEALTHY  eligible for traffic
//	EJECTED  circuit open after FailureThreshold consecutive failures (passive,
//	         from real calls) or a failed active probe; no traffic until the
//	         cooldown passes. Cooldowns double per ejection (capped) and reset
//	         after a healthy period.
//	PROBING  cooldown over; the next passing /readyz probe makes it HEALTHY
//	         again (a half-open circuit that never risks a real request)
//
// Endpoints come from a resolver (DNS of a headless Service, or a fixed list)
// that is re-polled, so gateway pods can be added, removed or replaced.
package upstream

import (
	"context"
	"math/rand/v2"
	"net"
	"net/http"
	"slices"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
)

// Resolver returns the current endpoint addresses ("host:port").
type Resolver func(ctx context.Context) ([]string, error)

// StaticResolver always returns the same addresses.
func StaticResolver(addrs ...string) Resolver {
	return func(context.Context) ([]string, error) { return addrs, nil }
}

// DNSResolver resolves a (headless) Service name to one address per pod.
func DNSResolver(host string, port int) Resolver {
	return func(ctx context.Context) ([]string, error) {
		ips, err := net.DefaultResolver.LookupHost(ctx, host)
		if err != nil {
			return nil, err
		}
		out := make([]string, len(ips))
		for i, ip := range ips {
			out[i] = net.JoinHostPort(ip, strconv.Itoa(port))
		}
		sort.Strings(out)
		return out, nil
	}
}

type Config struct {
	ProbePath        string        // default /readyz
	ProbeInterval    time.Duration // default 1s
	ProbeTimeout     time.Duration // default 500ms
	ResolveInterval  time.Duration // default 5s
	FailureThreshold int           // consecutive failures before ejection (default 3)
	BaseEjection     time.Duration // first cooldown (default 2s)
	MaxEjection      time.Duration // cooldown cap (default 30s)
	// OnStateChange is called on every transition (metrics/logging; may be nil).
	OnStateChange func(addr string, from, to pb.UpstreamState)
}

func (c *Config) defaults() {
	if c.ProbePath == "" {
		c.ProbePath = "/readyz"
	}
	if c.ProbeInterval == 0 {
		c.ProbeInterval = time.Second
	}
	if c.ProbeTimeout == 0 {
		c.ProbeTimeout = 500 * time.Millisecond
	}
	if c.ResolveInterval == 0 {
		c.ResolveInterval = 5 * time.Second
	}
	if c.FailureThreshold == 0 {
		c.FailureThreshold = 3
	}
	if c.BaseEjection == 0 {
		c.BaseEjection = 2 * time.Second
	}
	if c.MaxEjection == 0 {
		c.MaxEjection = 30 * time.Second
	}
}

// Endpoint is one gateway.
type Endpoint struct {
	Addr     string
	inflight atomic.Int64

	mu           sync.Mutex
	state        pb.UpstreamState
	failures     int
	ejections    int64
	ejectedUntil time.Time
	healthySince time.Time
	lastSuccess  time.Time // last successful real call
	changedAt    time.Time // last state transition
}

func (e *Endpoint) Inflight() int64 { return e.inflight.Load() }

// Pool is the set of endpoints plus their health.
type Pool struct {
	cfg     Config
	resolve Resolver
	client  *http.Client
	now     func() time.Time

	mu        sync.RWMutex
	endpoints map[string]*Endpoint
}

func NewPool(resolve Resolver, cfg Config) *Pool {
	cfg.defaults()
	return &Pool{
		cfg: cfg, resolve: resolve, now: time.Now,
		client:    &http.Client{Timeout: cfg.ProbeTimeout},
		endpoints: map[string]*Endpoint{},
	}
}

// Run resolves and probes until ctx ends. Call Refresh once first to start warm.
func (p *Pool) Run(ctx context.Context) {
	probe := time.NewTicker(p.cfg.ProbeInterval)
	resolve := time.NewTicker(p.cfg.ResolveInterval)
	defer probe.Stop()
	defer resolve.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-resolve.C:
			_ = p.Refresh(ctx)
		case <-probe.C:
			p.ProbeAll(ctx)
		}
	}
}

// Refresh re-resolves endpoints: new ones start PROBING (they get traffic
// only after passing a probe), vanished ones are dropped.
func (p *Pool) Refresh(ctx context.Context) error {
	addrs, err := p.resolve(ctx)
	if err != nil {
		return err // keep the last known set
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	keep := map[string]bool{}
	for _, a := range addrs {
		keep[a] = true
		if _, ok := p.endpoints[a]; !ok {
			p.endpoints[a] = &Endpoint{Addr: a, state: pb.UpstreamState_UPSTREAM_STATE_PROBING}
		}
	}
	for a := range p.endpoints {
		if !keep[a] {
			delete(p.endpoints, a)
		}
	}
	return nil
}

// ProbeAll runs one active health probe per endpoint, concurrently.
func (p *Pool) ProbeAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, e := range p.All() {
		wg.Add(1)
		go func(e *Endpoint) {
			defer wg.Done()
			p.probe(ctx, e)
		}(e)
	}
	wg.Wait()
}

func (p *Pool) probe(ctx context.Context, e *Endpoint) {
	e.mu.Lock()
	if e.state == pb.UpstreamState_UPSTREAM_STATE_EJECTED {
		if p.now().Before(e.ejectedUntil) {
			e.mu.Unlock()
			return // still cooling down
		}
		p.set(e, pb.UpstreamState_UPSTREAM_STATE_PROBING) // half-open
	}
	e.mu.Unlock()
	ok := false
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+e.Addr+p.cfg.ProbePath, nil)
	if err == nil {
		if resp, err := p.client.Do(req); err == nil {
			resp.Body.Close()
			ok = resp.StatusCode == http.StatusOK
		}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	switch {
	case ok && e.state != pb.UpstreamState_UPSTREAM_STATE_HEALTHY:
		p.set(e, pb.UpstreamState_UPSTREAM_STATE_HEALTHY)
		e.failures = 0
		e.healthySince = p.now()
	case ok && p.now().Sub(e.healthySince) > p.cfg.MaxEjection:
		e.ejections = 0 // healthy for a while: next ejection starts short again
	case !ok:
		p.eject(e) // a failed probe (healthy, new, or half-open) ejects, each time for longer
	}
}

// must hold e.mu
func (p *Pool) eject(e *Endpoint) {
	cooldown := p.cfg.BaseEjection << min(e.ejections, 8)
	cooldown = min(cooldown, p.cfg.MaxEjection)
	e.ejections++
	e.ejectedUntil = p.now().Add(cooldown)
	p.set(e, pb.UpstreamState_UPSTREAM_STATE_EJECTED)
}

// must hold e.mu
func (p *Pool) set(e *Endpoint, to pb.UpstreamState) {
	from := e.state
	e.state = to
	if from != to {
		e.changedAt = p.now()
	}
	if from != to && p.cfg.OnStateChange != nil {
		p.cfg.OnStateChange(e.Addr, from, to)
	}
}

// All returns every known endpoint (sorted by address).
func (p *Pool) All() []*Endpoint {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]*Endpoint, 0, len(p.endpoints))
	for _, e := range p.endpoints {
		out = append(out, e)
	}
	slices.SortFunc(out, func(a, b *Endpoint) int {
		if a.Addr < b.Addr {
			return -1
		}
		return 1
	})
	return out
}

func (e *Endpoint) healthy() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.state == pb.UpstreamState_UPSTREAM_STATE_HEALTHY
}

// Pick returns a healthy endpoint not in exclude, choosing the less loaded of
// two random candidates (power of two choices), or nil if none is healthy.
func (p *Pool) Pick(exclude map[string]bool) *Endpoint {
	var cands []*Endpoint
	for _, e := range p.All() {
		if !exclude[e.Addr] && e.healthy() {
			cands = append(cands, e)
		}
	}
	switch len(cands) {
	case 0:
		return nil
	case 1:
		return cands[0]
	}
	i, j := rand.IntN(len(cands)), rand.IntN(len(cands)-1)
	if j >= i {
		j++
	}
	if cands[j].Inflight() < cands[i].Inflight() {
		return cands[j]
	}
	return cands[i]
}

// Begin marks a call as started on e; call the returned func with the outcome.
func (p *Pool) Begin(e *Endpoint) func(success bool) {
	e.inflight.Add(1)
	return func(success bool) {
		e.inflight.Add(-1)
		e.mu.Lock()
		defer e.mu.Unlock()
		if success {
			e.failures = 0
			e.lastSuccess = p.now()
			return
		}
		e.failures++
		if e.failures >= p.cfg.FailureThreshold && e.state == pb.UpstreamState_UPSTREAM_STATE_HEALTHY {
			p.eject(e)
		}
	}
}

// Observations reports every endpoint as this node sees it (for heartbeats
// and GetClusterState).
func (p *Pool) Observations(observer string) []*pb.EndpointObservation {
	var out []*pb.EndpointObservation
	for _, e := range p.All() {
		e.mu.Lock()
		at := e.changedAt
		if at.IsZero() {
			at = p.now()
		}
		out = append(out, &pb.EndpointObservation{
			Address: e.Addr, State: e.state, ObservedUnixMs: at.UnixMilli(),
			Ejections: e.ejections, Inflight: e.Inflight(), Observer: observer,
		})
		e.mu.Unlock()
	}
	return out
}

// Snapshot is Observations without an observer name.
func (p *Pool) Snapshot() []*pb.EndpointObservation { return p.Observations("") }

// EjectFromPeer applies a peer's fresh ejection of addr: if this node still
// routes to it and has not seen it succeed since the peer's observation, it
// stops routing there for one base cooldown, after which its own probe
// decides. A peer can only take an endpoint out of rotation, never put one
// back: re-admission always needs this node's own passing probe.
func (p *Pool) EjectFromPeer(addr string, observedAt time.Time) bool {
	p.mu.RLock()
	e, ok := p.endpoints[addr]
	p.mu.RUnlock()
	if !ok {
		return false
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.state != pb.UpstreamState_UPSTREAM_STATE_HEALTHY || !e.lastSuccess.Before(observedAt) {
		return false
	}
	e.ejectedUntil = p.now().Add(p.cfg.BaseEjection)
	p.set(e, pb.UpstreamState_UPSTREAM_STATE_EJECTED)
	return true
}

// Healthy counts healthy endpoints.
func (p *Pool) Healthy() int {
	n := 0
	for _, e := range p.All() {
		if e.healthy() {
			n++
		}
	}
	return n
}
