package controlplane_test

// Multi-node integration test: three real control-plane nodes (Router +
// Cluster) on TCP behind one failover client, two fake gateways, continuous
// closed-loop Search traffic plus Chat turns. Mid-traffic one node crashes
// (connections dropped, heartbeats stop, like SIGKILL), one drains (SIGTERM
// path) and one gateway dies. Every Search must succeed and every Chat must
// complete with an intact answer.

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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
	"google.golang.org/grpc/status"

	"github.com/JCHETAN26/Omni-agent/controlplane/client"
	pb "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/cluster"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/server"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/upstream"
)

var words = []string{"Check", " the", " RAID", " controller", " logs", " first."}

func fakeGateway(name string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/search":
			time.Sleep(2 * time.Millisecond)
			json.NewEncoder(w).Encode(map[string]any{"hits": []any{map[string]any{
				"chunk": map[string]any{"chunk_id": "c", "doc_id": name}, "score": 1}}})
		case "/sse/agent":
			fl := w.(http.Flusher)
			for i, tok := range words {
				b, _ := json.Marshal(map[string]any{"type": "response.token", "index": i, "text": tok})
				fmt.Fprintf(w, "data: %s\n\n", b)
				fl.Flush()
				time.Sleep(15 * time.Millisecond) // a turn spans the faults below
			}
			sum := sha256.Sum256([]byte(strings.Join(words, "")))
			b, _ := json.Marshal(map[string]any{"type": "response.end", "outcome": "answered",
				"tokens": len(words), "sha256": hex.EncodeToString(sum[:])})
			fmt.Fprintf(w, "data: %s\n\n", b)
		}
	}))
}

type node struct {
	srv   *server.Server
	g     *grpc.Server
	addr  string
	crash func()
}

func startNodes(t *testing.T, gateways []string) []*node {
	t.Helper()
	var lis []net.Listener
	var addrs []string
	for range 3 {
		l, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		lis, addrs = append(lis, l), append(addrs, l.Addr().String())
	}
	var nodes []*node
	for i := range 3 {
		ctx, cancel := context.WithCancel(context.Background())
		t.Cleanup(cancel)
		log := slog.New(slog.NewTextHandler(io.Discard, nil))
		pool := upstream.NewPool(upstream.StaticResolver(gateways...), upstream.Config{
			ProbeInterval: 50 * time.Millisecond, FailureThreshold: 1})
		_ = pool.Refresh(ctx)
		pool.ProbeAll(ctx)
		go pool.Run(ctx)
		self := &pb.NodeInfo{Id: fmt.Sprintf("n%d", i+1), Address: addrs[i]}
		srv := server.New(pool, server.Config{Node: self, PerTryTimeout: 500 * time.Millisecond}, log)
		members := cluster.New(cluster.Config{Self: self, Resolve: upstream.StaticResolver(addrs...),
			HeartbeatInterval: 100 * time.Millisecond}, pool, srv.Status, log)
		g := grpc.NewServer()
		pb.RegisterRouterServer(g, srv)
		pb.RegisterClusterServer(g, members)
		healthpb.RegisterHealthServer(g, srv.Health)
		go g.Serve(lis[i])
		go members.Run(ctx)
		t.Cleanup(g.Stop)
		nodes = append(nodes, &node{srv, g, addrs[i], func() { cancel(); g.Stop() }})
	}
	return nodes
}

func TestZeroFailedCallsThroughNodeCrashDrainAndGatewayLoss(t *testing.T) {
	gw1, gw2 := fakeGateway("gw1"), fakeGateway("gw2")
	defer gw2.Close()
	nodes := startNodes(t, []string{strings.TrimPrefix(gw1.URL, "http://"), strings.TrimPrefix(gw2.URL, "http://")})

	r := manual.NewBuilderWithScheme("test")
	var addrs []resolver.Address
	for _, n := range nodes {
		addrs = append(addrs, resolver.Address{Addr: n.addr})
	}
	r.InitialState(resolver.State{Addresses: addrs})
	conn, router, err := client.Dial(r.Scheme()+":///cp", grpc.WithResolvers(r))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	var ok, failed, chats, restarts atomic.Int64
	servedBy := sync.Map{}
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for range 8 { // closed-loop Search workers
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
				resp, err := router.Search(ctx, &pb.SearchRequest{Query: "q"})
				cancel()
				if err != nil {
					failed.Add(1)
					t.Logf("search failed: %v", status.Convert(err).Message())
					continue
				}
				ok.Add(1)
				n, _ := servedBy.LoadOrStore(resp.Route.Node, new(atomic.Int64))
				n.(*atomic.Int64).Add(1)
			}
		}()
	}
	for range 4 { // Chat turns, each long enough to straddle the faults
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				var text strings.Builder
				done, err := client.Chat(ctx, router, &pb.ChatRequest{Question: "why EIO?"}, 3, func(ev *pb.ChatEvent) {
					switch {
					case ev.GetRestart() != nil:
						text.Reset()
						restarts.Add(1)
					case ev.GetToken() != nil:
						text.WriteString(ev.GetToken().Text)
					}
				})
				cancel()
				sum := sha256.Sum256([]byte(text.String()))
				if err != nil || done.GetDone().Sha256 != hex.EncodeToString(sum[:]) {
					failed.Add(1)
					t.Logf("chat failed: err=%v text=%q", err, text.String())
					continue
				}
				chats.Add(1)
			}
		}()
	}
	time.Sleep(400 * time.Millisecond)
	nodes[0].crash() // SIGKILL-like: connections reset mid-call, heartbeats stop
	time.Sleep(400 * time.Millisecond)
	nodes[1].srv.Drain() // SIGTERM path: NOT_SERVING, then stop
	time.Sleep(200 * time.Millisecond)
	nodes[1].g.GracefulStop()
	time.Sleep(200 * time.Millisecond)
	gw1.Close() // a gateway dies too
	time.Sleep(500 * time.Millisecond)
	close(stop)
	wg.Wait()

	if failed.Load() != 0 {
		t.Fatalf("%d calls failed (%d searches ok, %d chats ok)", failed.Load(), ok.Load(), chats.Load())
	}
	if ok.Load() < 200 || chats.Load() < 8 {
		t.Fatalf("too little traffic: %d searches, %d chats", ok.Load(), chats.Load())
	}
	counts := map[string]int64{}
	servedBy.Range(func(k, v any) bool { counts[k.(string)] = v.(*atomic.Int64).Load(); return true })
	if len(counts) != 3 {
		t.Fatalf("traffic was not spread across all nodes before the faults: %v", counts)
	}
	t.Logf("%d searches + %d chats, 0 failed (%d chat restarts); served by %v",
		ok.Load(), chats.Load(), restarts.Load(), counts)
}
