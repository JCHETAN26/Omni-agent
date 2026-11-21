package controller

import (
	"context"
	"errors"
	"math"
	"net"
	"strconv"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	corev1 "k8s.io/api/core/v1"

	cpv1 "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
)

// recommendation is one autoscaling sample.
type recommendation struct {
	at       time.Time
	replicas int32
}

// decide returns the gateway replica count for the "operator" mode.
//
// Scale-up is immediate, to ceil(inflight / target). A scale-down is only
// recommended when the smaller count would carry the load at no more than
// headroom (e.g. 0.7) of the target: ceil(inflight / (target*headroom)).
// Without that margin, capacity just added is shed again as soon as it has
// brought in-flight work down (observed on kind: 3 -> 2 mid-load). Scale-down
// then goes only to the highest recommendation within the stabilization
// window (the same rule the HPA uses). Everything is clamped to [min, max].
func decide(now time.Time, current, minR, maxR int32, inflight float64, targetPerReplica int32,
	headroom float64, history []recommendation, stabilization time.Duration) (int32, []recommendation) {
	minR = max(minR, 1)
	maxR = max(maxR, minR)
	target := float64(max(targetPerReplica, 1))
	if headroom <= 0 || headroom > 1 {
		headroom = 1
	}
	raw := int32(math.Ceil(inflight / target))
	if raw < current {
		raw = min(int32(math.Ceil(inflight/(target*headroom))), current)
	}
	raw = min(max(raw, minR), maxR)

	kept := history[:0:0]
	for _, r := range history {
		if now.Sub(r.at) < stabilization {
			kept = append(kept, r)
		}
	}
	kept = append(kept, recommendation{now, raw})
	if raw >= current {
		return raw, kept
	}
	highest := raw
	for _, r := range kept {
		highest = max(highest, r.replicas)
	}
	return min(max(highest, minR), maxR), kept
}

// LoadSource reports the control plane's load.
type LoadSource interface {
	// Load sums each node's inflight_avg_10s and rps over the given control-plane pods.
	Load(ctx context.Context, pods []corev1.Pod) (inflight, rps float64, err error)
}

// selfStatus picks the answering node's own entry from its cluster view.
func selfStatus(cs *cpv1.ClusterState) *cpv1.NodeStatus {
	for _, n := range cs.Nodes {
		if n.Node != nil && n.Node.Id == cs.NodeId {
			return n
		}
	}
	return &cpv1.NodeStatus{}
}

// GRPCLoad polls every ready control-plane pod's Cluster.GetClusterState RPC
// and sums each node's own load (not its peers', which would double count).
type GRPCLoad struct {
	mu    sync.Mutex
	conns map[string]*grpc.ClientConn
}

func (g *GRPCLoad) conn(addr string) (*grpc.ClientConn, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.conns == nil {
		g.conns = map[string]*grpc.ClientConn{}
	}
	if c, ok := g.conns[addr]; ok {
		return c, nil
	}
	c, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err == nil {
		g.conns[addr] = c
	}
	return c, err
}

func (g *GRPCLoad) forget(addr string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if c, ok := g.conns[addr]; ok {
		_ = c.Close()
		delete(g.conns, addr)
	}
}

func (g *GRPCLoad) Load(ctx context.Context, pods []corev1.Pod) (float64, float64, error) {
	var (
		wg              sync.WaitGroup
		mu              sync.Mutex
		inflight, rps   float64
		errs            []error
		answered, tried int
	)
	for _, p := range pods {
		if p.Status.PodIP == "" || !podReady(&p) {
			continue
		}
		tried++
		addr := net.JoinHostPort(p.Status.PodIP, strconv.Itoa(grpcPort))
		wg.Add(1)
		go func() {
			defer wg.Done()
			cctx, cancel := context.WithTimeout(ctx, time.Second)
			defer cancel()
			c, err := g.conn(addr)
			var st *cpv1.NodeStatus
			if err == nil {
				var cs *cpv1.ClusterState
				cs, err = cpv1.NewClusterClient(c).GetClusterState(cctx, &cpv1.GetClusterStateRequest{})
				if err == nil {
					st = selfStatus(cs)
				}
			}
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				errs = append(errs, err)
				g.forget(addr)
				return
			}
			answered++
			inflight += st.InflightAvg_10S
			rps += st.Rps
		}()
	}
	wg.Wait()
	if tried > 0 && answered == 0 {
		return 0, 0, errors.Join(errs...)
	}
	return inflight, rps, nil
}

func podReady(p *corev1.Pod) bool {
	for _, c := range p.Status.Conditions {
		if c.Type == corev1.PodReady {
			return c.Status == corev1.ConditionTrue
		}
	}
	return false
}
