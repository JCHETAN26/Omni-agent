// omni-controlplane is one node of the gRPC control plane (omni.control.v1).
//
// Configuration (environment):
//
//	GRPC_PORT        9000 (Router, Cluster, grpc.health.v1, reflection)
//	HTTP_ADDR        :9090 (/metrics, /healthz)
//	GATEWAY_HOST     DNS name resolving to every gateway pod (a headless Service)
//	GATEWAY_PORT     8000
//	GATEWAY_ADDRS    comma-separated host:port list instead of GATEWAY_HOST
//	PEER_HOST        DNS name resolving to every control-plane pod (headless Service)
//	PEER_ADDRS       comma-separated host:port list instead of PEER_HOST
//	POD_IP           this node's address as peers see it (default: first non-loopback IP)
//	NODE_NAME        node identity (default: hostname)
//	ELASTICSEARCH_URL, ELASTICSEARCH_INDEX (omniagent_chunks), ELASTICSEARCH_API_KEY
//	                 enable the BM25 degraded mode
//	DRAIN_SECONDS    on SIGTERM, report NOT_SERVING this long before stopping (5)
//	PER_TRY_TIMEOUT_MS (2000), MAX_ATTEMPTS (3), DEFAULT_TIMEOUT_MS (5000)
//	HEARTBEAT_MS (500), DEAD_AFTER (3 missed heartbeats)
package main

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/reflection"

	pb "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/cluster"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/metrics"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/server"
	"github.com/JCHETAN26/Omni-agent/controlplane/internal/upstream"
)

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return def
}

func resolver(listEnv, hostEnv string, port int) upstream.Resolver {
	if addrs := os.Getenv(listEnv); addrs != "" {
		return upstream.StaticResolver(strings.Split(addrs, ",")...)
	}
	if host := os.Getenv(hostEnv); host != "" {
		return upstream.DNSResolver(host, port)
	}
	return nil
}

func localIP() string {
	if ip := os.Getenv("POD_IP"); ip != "" {
		return ip
	}
	addrs, _ := net.InterfaceAddrs()
	for _, a := range addrs {
		if n, ok := a.(*net.IPNet); ok && !n.IP.IsLoopback() && n.IP.To4() != nil {
			return n.IP.String()
		}
	}
	return "127.0.0.1"
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil)).With("service", "omni-controlplane")
	host, _ := os.Hostname()
	port := envInt("GRPC_PORT", 9000)
	self := &pb.NodeInfo{
		Id:            env("NODE_NAME", host),
		Address:       net.JoinHostPort(localIP(), strconv.Itoa(port)),
		StartedUnixMs: time.Now().UnixMilli(),
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	gateways := resolver("GATEWAY_ADDRS", "GATEWAY_HOST", envInt("GATEWAY_PORT", 8000))
	if gateways == nil {
		gateways = upstream.StaticResolver("localhost:8000")
	}
	pool := upstream.NewPool(gateways, upstream.Config{
		OnStateChange: func(addr string, from, to pb.UpstreamState) {
			metrics.Transitions.WithLabelValues(to.String()).Inc()
			log.Info("gateway state", "addr", addr, "from", from.String(), "to", to.String(),
				"unix_ms", time.Now().UnixMilli())
		},
	})
	if err := pool.Refresh(ctx); err != nil {
		log.Warn("initial gateway resolution failed; will retry", "err", err)
	}
	pool.ProbeAll(ctx)
	go pool.Run(ctx)
	go func() {
		for range time.Tick(time.Second) {
			metrics.Healthy.Set(float64(pool.Healthy()))
		}
	}()

	var bm25 *server.BM25
	if u := os.Getenv("ELASTICSEARCH_URL"); u != "" {
		bm25 = server.NewBM25(u, env("ELASTICSEARCH_INDEX", "omniagent_chunks"), os.Getenv("ELASTICSEARCH_API_KEY"))
	}
	srv := server.New(pool, server.Config{
		Node:           self,
		DefaultTimeout: time.Duration(envInt("DEFAULT_TIMEOUT_MS", 5000)) * time.Millisecond,
		PerTryTimeout:  time.Duration(envInt("PER_TRY_TIMEOUT_MS", 2000)) * time.Millisecond,
		MaxAttempts:    envInt("MAX_ATTEMPTS", 3),
		BM25:           bm25,
	}, log)
	go srv.SampleInflight(ctx)

	peers := resolver("PEER_ADDRS", "PEER_HOST", port)
	if peers == nil {
		peers = upstream.StaticResolver()
	}
	members := cluster.New(cluster.Config{
		Self: self, Resolve: peers,
		HeartbeatInterval: time.Duration(envInt("HEARTBEAT_MS", 500)) * time.Millisecond,
		DeadAfter:         envInt("DEAD_AFTER", 3),
	}, pool, srv.Status, log)

	g := grpc.NewServer(
		grpc.KeepaliveParams(keepalive.ServerParameters{Time: 30 * time.Second}),
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{MinTime: 10 * time.Second, PermitWithoutStream: true}),
	)
	pb.RegisterRouterServer(g, srv)
	pb.RegisterClusterServer(g, members)
	healthpb.RegisterHealthServer(g, srv.Health)
	reflection.Register(g)

	lis, err := net.Listen("tcp", ":"+strconv.Itoa(port))
	if err != nil {
		log.Error("listen", "err", err)
		os.Exit(1)
	}
	mux := http.NewServeMux()
	mux.Handle("GET /metrics", promhttp.Handler())
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok\n")) })
	hs := &http.Server{Addr: env("HTTP_ADDR", ":9090"), Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		if err := hs.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("metrics server", "err", err)
		}
	}()
	go func() {
		if err := g.Serve(lis); err != nil {
			log.Error("grpc server", "err", err)
			os.Exit(1)
		}
	}()
	go members.Run(ctx)
	log.Info("control plane node started", "node", self.Id, "address", self.Address,
		"gateways", len(pool.All()), "bm25_fallback", bm25 != nil)

	<-ctx.Done()
	// Graceful shutdown: NOT_SERVING (clients and peers see DRAINING), let
	// in-flight calls finish, then stop.
	srv.Drain()
	time.Sleep(time.Duration(envInt("DRAIN_SECONDS", 5)) * time.Second)
	done := make(chan struct{})
	go func() { g.GracefulStop(); close(done) }()
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		g.Stop()
	}
	shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = hs.Shutdown(shutdown)
	log.Info("control plane node stopped", "node", self.Id)
}
