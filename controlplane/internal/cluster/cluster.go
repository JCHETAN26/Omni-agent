// Package cluster gives control-plane nodes peer awareness.
//
// Every node knows its peers (DNS of the headless Service, re-resolved), and
// registers with them at startup. Each HeartbeatInterval it exchanges a
// Heartbeat with every peer, carrying its status and its gateway table.
//
//   - A peer that misses one heartbeat is SUSPECT, DeadAfter in a row DEAD
//     (logged with the detection time, counted, and timed in a histogram).
//     A peer reporting DRAINING is shown as such. Recovery is logged too.
//   - Gateway ejections a peer observed are applied locally (see
//     upstream.Pool.EjectFromPeer), so one node's detection of a failed
//     gateway takes it out of rotation on every node within a heartbeat.
//
// Clients fail over between nodes on their own (health checks + retries, see
// package client); membership is for detection, observability and sharing
// upstream health, not for forwarding.
package cluster

import (
	"context"
	"log/slog"
	"sort"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/metrics"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/upstream"
)

type Config struct {
	Self              *pb.NodeInfo
	Resolve           upstream.Resolver // peer gRPC addresses; may include Self.Address
	HeartbeatInterval time.Duration     // default 500ms
	HeartbeatTimeout  time.Duration     // default 300ms
	DeadAfter         int               // consecutive misses before DEAD (default 3)
	ResolveInterval   time.Duration     // default 2s
}

type peer struct {
	addr   string
	status *pb.NodeStatus
	state  pb.NodeState
	missed int
	lastOK time.Time
	conn   *grpc.ClientConn
}

// Membership is this node's view of its peers; it also serves the Cluster RPCs.
type Membership struct {
	pb.UnimplementedClusterServer
	cfg    Config
	pool   *upstream.Pool
	status func() *pb.NodeStatus // this node's live status
	log    *slog.Logger
	now    func() time.Time

	mu    sync.Mutex
	peers map[string]*peer
}

func New(cfg Config, pool *upstream.Pool, status func() *pb.NodeStatus, log *slog.Logger) *Membership {
	if cfg.HeartbeatInterval == 0 {
		cfg.HeartbeatInterval = 500 * time.Millisecond
	}
	if cfg.HeartbeatTimeout == 0 {
		cfg.HeartbeatTimeout = 300 * time.Millisecond
	}
	if cfg.DeadAfter == 0 {
		cfg.DeadAfter = 3
	}
	if cfg.ResolveInterval == 0 {
		cfg.ResolveInterval = 2 * time.Second
	}
	return &Membership{cfg: cfg, pool: pool, status: status, log: log, now: time.Now, peers: map[string]*peer{}}
}

// Run resolves peers, registers with them, and heartbeats until ctx ends.
func (m *Membership) Run(ctx context.Context) {
	m.resolve(ctx)
	m.registerAll(ctx)
	hb := time.NewTicker(m.cfg.HeartbeatInterval)
	rs := time.NewTicker(m.cfg.ResolveInterval)
	defer hb.Stop()
	defer rs.Stop()
	for {
		select {
		case <-ctx.Done():
			m.mu.Lock()
			for _, p := range m.peers {
				if p.conn != nil {
					p.conn.Close()
				}
			}
			m.mu.Unlock()
			return
		case <-rs.C:
			m.resolve(ctx)
		case <-hb.C:
			m.HeartbeatAll(ctx)
		}
	}
}

func (m *Membership) resolve(ctx context.Context) {
	addrs, err := m.cfg.Resolve(ctx)
	if err != nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	seen := map[string]bool{}
	for _, a := range addrs {
		if a == m.cfg.Self.Address {
			continue
		}
		seen[a] = true
		m.addLocked(a)
	}
	// Forget peers that left DNS and are no longer answering (replaced pods).
	for a, p := range m.peers {
		if !seen[a] && p.state == pb.NodeState_NODE_STATE_DEAD {
			if p.conn != nil {
				p.conn.Close()
			}
			delete(m.peers, a)
		}
	}
}

// must hold m.mu
func (m *Membership) addLocked(addr string) *peer {
	if p, ok := m.peers[addr]; ok {
		return p
	}
	p := &peer{addr: addr, state: pb.NodeState_NODE_STATE_SUSPECT}
	m.peers[addr] = p
	return p
}

func (m *Membership) client(p *peer) (pb.ClusterClient, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if p.conn == nil {
		c, err := grpc.NewClient(p.addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			return nil, err
		}
		p.conn = c
	}
	return pb.NewClusterClient(p.conn), nil
}

func (m *Membership) list() []*peer {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*peer, 0, len(m.peers))
	for _, p := range m.peers {
		out = append(out, p)
	}
	return out
}

func (m *Membership) registerAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, p := range m.list() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := m.client(p)
			if err != nil {
				return
			}
			cctx, cancel := context.WithTimeout(ctx, m.cfg.HeartbeatTimeout)
			defer cancel()
			if _, err := c.Register(cctx, &pb.RegisterRequest{Node: m.cfg.Self}); err == nil {
				m.markAlive(p, nil)
			}
		}()
	}
	wg.Wait()
}

// HeartbeatAll exchanges one heartbeat with every peer, concurrently.
func (m *Membership) HeartbeatAll(ctx context.Context) {
	req := &pb.HeartbeatRequest{Sender: m.status(), Endpoints: m.pool.Observations(m.cfg.Self.Id)}
	var wg sync.WaitGroup
	for _, p := range m.list() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := m.client(p)
			var resp *pb.HeartbeatResponse
			if err == nil {
				cctx, cancel := context.WithTimeout(ctx, m.cfg.HeartbeatTimeout)
				resp, err = c.Heartbeat(cctx, req)
				cancel()
			}
			if err != nil {
				m.markMissed(p)
				return
			}
			m.markAlive(p, resp.Sender)
			m.apply(resp.Endpoints)
		}()
	}
	wg.Wait()
}

func (m *Membership) markAlive(p *peer, st *pb.NodeStatus) {
	m.mu.Lock()
	defer m.mu.Unlock()
	was := p.state
	p.missed, p.lastOK = 0, m.now()
	if st != nil {
		p.status = st
	}
	p.state = pb.NodeState_NODE_STATE_ALIVE
	if st != nil && st.State == pb.NodeState_NODE_STATE_DRAINING {
		p.state = pb.NodeState_NODE_STATE_DRAINING
	}
	if was == pb.NodeState_NODE_STATE_DEAD {
		metrics.PeerEvents.WithLabelValues("recovered").Inc()
		m.log.Info("peer recovered", "peer", p.addr)
	}
	metrics.PeersAlive.Set(float64(m.aliveLocked()))
}

func (m *Membership) markMissed(p *peer) {
	m.mu.Lock()
	defer m.mu.Unlock()
	p.missed++
	switch {
	case p.state == pb.NodeState_NODE_STATE_DEAD:
	case p.missed >= m.cfg.DeadAfter:
		p.state = pb.NodeState_NODE_STATE_DEAD
		detect := time.Duration(0)
		if !p.lastOK.IsZero() {
			detect = m.now().Sub(p.lastOK)
			metrics.PeerDetectSeconds.Observe(detect.Seconds())
		}
		metrics.PeerEvents.WithLabelValues("dead").Inc()
		m.log.Warn("peer down", "peer", p.addr, "missed", p.missed,
			"since_last_heartbeat_ms", detect.Milliseconds(), "detected_unix_ms", m.now().UnixMilli())
	default:
		p.state = pb.NodeState_NODE_STATE_SUSPECT
	}
	metrics.PeersAlive.Set(float64(m.aliveLocked()))
}

// must hold m.mu
func (m *Membership) aliveLocked() int {
	n := 0
	for _, p := range m.peers {
		if p.state == pb.NodeState_NODE_STATE_ALIVE {
			n++
		}
	}
	return n
}

// apply takes a peer's gateway observations: its fresh ejections become ours.
func (m *Membership) apply(obs []*pb.EndpointObservation) {
	for _, o := range obs {
		if o.State == pb.UpstreamState_UPSTREAM_STATE_EJECTED &&
			m.pool.EjectFromPeer(o.Address, time.UnixMilli(o.ObservedUnixMs)) {
			metrics.PeerEjections.Inc()
			m.log.Info("gateway ejected on a peer's report", "gateway", o.Address, "observer", o.Observer)
		}
	}
}

// --- Cluster service ---------------------------------------------------------

func (m *Membership) Register(_ context.Context, req *pb.RegisterRequest) (*pb.ClusterState, error) {
	if req.Node != nil && req.Node.Address != "" && req.Node.Address != m.cfg.Self.Address {
		m.mu.Lock()
		p := m.addLocked(req.Node.Address)
		m.mu.Unlock()
		m.markAlive(p, &pb.NodeStatus{Node: req.Node, State: pb.NodeState_NODE_STATE_ALIVE})
		m.log.Info("peer registered", "peer", req.Node.Address, "id", req.Node.Id)
	}
	return m.State(), nil
}

func (m *Membership) Heartbeat(_ context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	if s := req.Sender; s != nil && s.Node != nil && s.Node.Address != m.cfg.Self.Address {
		m.mu.Lock()
		p := m.addLocked(s.Node.Address)
		m.mu.Unlock()
		m.markAlive(p, s)
		m.apply(req.Endpoints)
	}
	return &pb.HeartbeatResponse{Sender: m.status(), Endpoints: m.pool.Observations(m.cfg.Self.Id)}, nil
}

func (m *Membership) GetClusterState(context.Context, *pb.GetClusterStateRequest) (*pb.ClusterState, error) {
	return m.State(), nil
}

// State is this node's view: itself plus every known peer.
func (m *Membership) State() *pb.ClusterState {
	nodes := []*pb.NodeStatus{m.status()}
	m.mu.Lock()
	for _, p := range m.peers {
		st := &pb.NodeStatus{Node: &pb.NodeInfo{Address: p.addr}}
		if p.status != nil {
			st = &pb.NodeStatus{Node: p.status.Node, Inflight: p.status.Inflight, InflightAvg_10S: p.status.InflightAvg_10S,
				Rps: p.status.Rps, RequestsTotal: p.status.RequestsTotal}
		}
		st.State = p.state
		if !p.lastOK.IsZero() {
			st.LastHeartbeatUnixMs = p.lastOK.UnixMilli()
		}
		nodes = append(nodes, st)
	}
	m.mu.Unlock()
	sort.Slice(nodes[1:], func(i, j int) bool { return nodes[i+1].Node.Address < nodes[j+1].Node.Address })
	return &pb.ClusterState{NodeId: m.cfg.Self.Id, Nodes: nodes, Endpoints: m.pool.Observations(m.cfg.Self.Id)}
}

// PeerState returns a peer's state by address (tests and debugging).
func (m *Membership) PeerState(addr string) pb.NodeState {
	m.mu.Lock()
	defer m.mu.Unlock()
	if p, ok := m.peers[addr]; ok {
		return p.state
	}
	return pb.NodeState_NODE_STATE_UNSPECIFIED
}
