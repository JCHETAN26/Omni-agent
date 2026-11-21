package upstream

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	pb "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
)

type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time { return c.t }

func gateway(ready *atomic.Bool) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if !ready.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	}))
}

func addr(s *httptest.Server) string { return strings.TrimPrefix(s.URL, "http://") }

func state(p *Pool, a string) pb.UpstreamState {
	for _, u := range p.Snapshot() {
		if u.Address == a {
			return u.State
		}
	}
	return pb.UpstreamState_UPSTREAM_STATE_UNSPECIFIED
}

func TestNewEndpointsNeedAPassingProbe(t *testing.T) {
	var ready atomic.Bool
	g := gateway(&ready)
	defer g.Close()
	p := NewPool(StaticResolver(addr(g)), Config{})
	ctx := context.Background()
	_ = p.Refresh(ctx)
	if p.Pick(nil) != nil {
		t.Fatal("an unprobed endpoint must not get traffic")
	}
	ready.Store(true)
	p.ProbeAll(ctx)
	if p.Pick(nil) == nil {
		t.Fatal("a probed endpoint must get traffic")
	}
}

func TestPassiveEjectionAndHalfOpenRecovery(t *testing.T) {
	var ready atomic.Bool
	ready.Store(true)
	g := gateway(&ready)
	defer g.Close()
	clock := &fakeClock{time.Unix(1000, 0)}
	var transitions []string
	p := NewPool(StaticResolver(addr(g)), Config{FailureThreshold: 3, BaseEjection: 2 * time.Second,
		OnStateChange: func(_ string, _, to pb.UpstreamState) { transitions = append(transitions, to.String()) }})
	p.now = clock.now
	ctx := context.Background()
	_ = p.Refresh(ctx)
	p.ProbeAll(ctx)
	e := p.Pick(nil)
	for range 2 {
		p.Begin(e)(false)
	}
	if state(p, e.Addr) != pb.UpstreamState_UPSTREAM_STATE_HEALTHY {
		t.Fatal("ejected before the threshold")
	}
	p.Begin(e)(false)
	if state(p, e.Addr) != pb.UpstreamState_UPSTREAM_STATE_EJECTED || p.Pick(nil) != nil {
		t.Fatal("3 consecutive failures must eject")
	}
	p.ProbeAll(ctx) // still cooling down: a passing probe must not readmit it
	if p.Pick(nil) != nil {
		t.Fatal("readmitted during cooldown")
	}
	clock.t = clock.t.Add(2 * time.Second)
	p.ProbeAll(ctx)
	if p.Pick(nil) == nil {
		t.Fatal("not readmitted after cooldown + passing probe")
	}
	want := "UPSTREAM_STATE_HEALTHY,UPSTREAM_STATE_EJECTED,UPSTREAM_STATE_PROBING,UPSTREAM_STATE_HEALTHY"
	if got := strings.Join(transitions, ","); got != want {
		t.Fatalf("transitions %s", got)
	}
}

func TestCooldownGrowsWhileStillFailing(t *testing.T) {
	var ready atomic.Bool
	g := gateway(&ready)
	defer g.Close()
	clock := &fakeClock{time.Unix(1000, 0)}
	p := NewPool(StaticResolver(addr(g)), Config{BaseEjection: time.Second, MaxEjection: 4 * time.Second})
	p.now = clock.now
	ctx := context.Background()
	_ = p.Refresh(ctx)
	var cooldowns []time.Duration
	for range 4 {
		p.ProbeAll(ctx)
		e := p.All()[0]
		cooldowns = append(cooldowns, e.ejectedUntil.Sub(clock.t))
		clock.t = e.ejectedUntil
	}
	want := []time.Duration{time.Second, 2 * time.Second, 4 * time.Second, 4 * time.Second}
	for i := range want {
		if cooldowns[i] != want[i] {
			t.Fatalf("cooldowns %v, want %v", cooldowns, want)
		}
	}
}

func TestRefreshTracksMembershipAndPickBalances(t *testing.T) {
	var ready atomic.Bool
	ready.Store(true)
	a, b := gateway(&ready), gateway(&ready)
	defer a.Close()
	defer b.Close()
	addrs := []string{addr(a), addr(b)}
	p := NewPool(func(context.Context) ([]string, error) { return addrs, nil }, Config{})
	ctx := context.Background()
	_ = p.Refresh(ctx)
	p.ProbeAll(ctx)
	busy := p.All()[0]
	release := p.Begin(busy)
	for range 20 {
		if got := p.Pick(nil); got == busy {
			t.Fatal("power of two choices must prefer the idle endpoint")
		}
	}
	release(true)
	if p.Pick(map[string]bool{addrs[0]: true, addrs[1]: true}) != nil {
		t.Fatal("exclusions ignored")
	}
	addrs = addrs[1:]
	_ = p.Refresh(ctx)
	if len(p.All()) != 1 {
		t.Fatal("removed endpoint kept")
	}
}
