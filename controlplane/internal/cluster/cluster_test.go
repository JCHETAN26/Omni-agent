package cluster

import (
	"context"
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

	pb "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/upstream"
)

type node struct {
	m     *Membership
	pool  *upstream.Pool
	g     *grpc.Server
	addr  string
	crash func() // like SIGKILL: stops serving and heartbeating at once
}

// cluster starts n nodes on localhost that know each other's addresses.
func cluster(t *testing.T, n int, gateways []string, interval time.Duration) []*node {
	t.Helper()
	var lis []net.Listener
	var addrs []string
	for range n {
		l, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		lis = append(lis, l)
		addrs = append(addrs, l.Addr().String())
	}
	var nodes []*node
	for i := range n {
		ctx, cancel := context.WithCancel(context.Background())
		t.Cleanup(cancel)
		pool := upstream.NewPool(upstream.StaticResolver(gateways...), upstream.Config{
			FailureThreshold: 1, ProbeInterval: time.Hour, BaseEjection: 200 * time.Millisecond})
		_ = pool.Refresh(ctx)
		pool.ProbeAll(ctx)
		self := &pb.NodeInfo{Id: addrs[i], Address: addrs[i]}
		m := New(Config{Self: self, Resolve: upstream.StaticResolver(addrs...), HeartbeatInterval: interval,
			HeartbeatTimeout: interval / 2, DeadAfter: 3},
			pool, func() *pb.NodeStatus { return &pb.NodeStatus{Node: self, State: pb.NodeState_NODE_STATE_ALIVE} },
			slog.New(slog.NewTextHandler(io.Discard, nil)))
		g := grpc.NewServer()
		pb.RegisterClusterServer(g, m)
		go g.Serve(lis[i])
		t.Cleanup(g.Stop)
		nodes = append(nodes, &node{m, pool, g, addrs[i], func() { cancel(); g.Stop() }})
		go m.Run(ctx)
	}
	return nodes
}

func waitFor(t *testing.T, what string, timeout time.Duration, cond func() bool) time.Duration {
	t.Helper()
	start := time.Now()
	for time.Since(start) < timeout {
		if cond() {
			return time.Since(start)
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
	return 0
}

func TestPeersDetectANodeFailureWithinThreeHeartbeats(t *testing.T) {
	interval := 100 * time.Millisecond
	nodes := cluster(t, 3, nil, interval)
	waitFor(t, "all peers alive", 2*time.Second, func() bool {
		for _, a := range nodes {
			for _, b := range nodes {
				if a != b && a.m.PeerState(b.addr) != pb.NodeState_NODE_STATE_ALIVE {
					return false
				}
			}
		}
		return true
	})
	nodes[0].crash()
	took := waitFor(t, "peers to declare the crashed node dead", 3*time.Second, func() bool {
		return nodes[1].m.PeerState(nodes[0].addr) == pb.NodeState_NODE_STATE_DEAD &&
			nodes[2].m.PeerState(nodes[0].addr) == pb.NodeState_NODE_STATE_DEAD
	})
	// 3 missed heartbeats at 100 ms, plus up to one interval of phase.
	if took > 600*time.Millisecond {
		t.Fatalf("detection took %v", took)
	}
	if st := nodes[1].m.State(); len(st.Nodes) != 3 {
		t.Fatalf("cluster state %v", st)
	}
	t.Logf("crashed node declared dead by both peers after %v", took)
}

func TestOneNodesGatewayEjectionReachesItsPeers(t *testing.T) {
	var down atomic.Bool
	gw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if down.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	}))
	defer gw.Close()
	addr := strings.TrimPrefix(gw.URL, "http://")
	nodes := cluster(t, 3, []string{addr}, 100*time.Millisecond)
	for _, nd := range nodes {
		if nd.pool.Healthy() != 1 {
			t.Fatal("gateway should start healthy everywhere")
		}
	}
	// Node 0 sees a real call fail; nodes 1 and 2 never call the gateway.
	down.Store(true)
	nodes[0].pool.Begin(nodes[0].pool.All()[0])(false)
	took := waitFor(t, "peers to eject the gateway", 2*time.Second, func() bool {
		return nodes[1].pool.Healthy() == 0 && nodes[2].pool.Healthy() == 0
	})
	t.Logf("ejection reached both peers in %v", took)

	// Re-admission is local only: once the gateway recovers, each node's own
	// probe (after the short peer-induced cooldown) brings it back.
	down.Store(false)
	time.Sleep(250 * time.Millisecond)
	for _, nd := range nodes {
		nd.pool.ProbeAll(context.Background())
	}
	for i, nd := range nodes {
		if nd.pool.Healthy() != 1 {
			t.Fatalf("node %d did not re-admit the recovered gateway", i)
		}
	}
}

func TestPeerCannotReadmitAGatewayThisNodeSeesFailing(t *testing.T) {
	gw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	defer gw.Close()
	addr := strings.TrimPrefix(gw.URL, "http://")
	nodes := cluster(t, 2, []string{addr}, time.Hour)
	nodes[0].pool.Begin(nodes[0].pool.All()[0])(false) // ejected locally
	// A peer reporting it HEALTHY must not put it back into rotation.
	nodes[0].m.apply([]*pb.EndpointObservation{{Address: addr, State: pb.UpstreamState_UPSTREAM_STATE_HEALTHY,
		ObservedUnixMs: time.Now().UnixMilli()}})
	if nodes[0].pool.Healthy() != 0 {
		t.Fatal("a peer's report re-admitted a locally ejected gateway")
	}
}
