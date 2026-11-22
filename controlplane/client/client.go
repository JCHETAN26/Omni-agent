// Package client dials the control plane with node failover built in.
//
// The target should resolve to every node, e.g. the headless Service
// "dns:///omniagent-controlplane:9000". The service config:
//   - round_robin across all resolved nodes
//   - client-side health checking (grpc.health.v1, "omni.control.v1.Router"):
//     a node that reports NOT_SERVING (draining) or stops answering leaves the
//     rotation
//   - unary calls (Search, Vision) are retried on UNAVAILABLE on another node
//     within the caller's deadline; both are idempotent. 5 attempts (gRPC's
//     maximum) with 0.1 s -> 1 s backoff give a crashed node's in-flight calls
//     ~1.5 s to find a live node; 4 attempts at 50-200 ms (~350 ms) occasionally
//     ran out on slow CI runners before the dead connections were noticed.
//
// Server-streaming Chat cannot be retried transparently once events have been
// received, so Chat (below) re-issues the turn on another node itself and
// tells the caller to discard partial output, the same contract as a gateway
// failure inside the control plane (a Restart event).
package client

import (
	"context"
	"errors"
	"fmt"
	"io"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	_ "google.golang.org/grpc/health" // registers client-side health checking
	"google.golang.org/grpc/status"

	pb "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
)

// ServiceConfig is the gRPC service config used by Dial.
const ServiceConfig = `{
  "loadBalancingConfig": [{"round_robin": {}}],
  "healthCheckConfig": {"serviceName": "omni.control.v1.Router"},
  "methodConfig": [{
    "name": [{"service": "omni.control.v1.Router"}, {"service": "omni.control.v1.Cluster"}],
    "waitForReady": true,
    "retryPolicy": {
      "maxAttempts": 5,
      "initialBackoff": "0.1s",
      "maxBackoff": "1s",
      "backoffMultiplier": 2,
      "retryableStatusCodes": ["UNAVAILABLE"]
    }
  }]
}`

// Dial connects to a control-plane target. In-cluster traffic is plaintext;
// pass extra options (e.g. TLS credentials) to override.
func Dial(target string, opts ...grpc.DialOption) (*grpc.ClientConn, pb.RouterClient, error) {
	base := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultServiceConfig(ServiceConfig),
	}
	conn, err := grpc.NewClient(target, append(base, opts...)...)
	if err != nil {
		return nil, nil, fmt.Errorf("dial %s: %w", target, err)
	}
	return conn, pb.NewRouterClient(conn), nil
}

// Chat runs one turn, calling onEvent for every event. If the node serving the
// stream dies mid-answer (UNAVAILABLE), the turn is re-issued (same request
// ID) up to maxRestarts times and onEvent first receives a Restart event.
// It returns the final Done event.
func Chat(ctx context.Context, c pb.RouterClient, req *pb.ChatRequest, maxRestarts int,
	onEvent func(*pb.ChatEvent)) (*pb.ChatEvent, error) {
	for attempt := 0; ; attempt++ {
		stream, err := c.Chat(ctx, req)
		if err == nil {
			var ev *pb.ChatEvent
			received := false
			for {
				ev, err = stream.Recv()
				if err != nil {
					break
				}
				received = true
				onEvent(ev)
				if ev.GetDone() != nil {
					return ev, nil
				}
			}
			if errors.Is(err, io.EOF) {
				err = status.Error(codes.Unavailable, "stream ended without a Done event")
			}
			if received && status.Code(err) == codes.Unavailable && attempt < maxRestarts && ctx.Err() == nil {
				onEvent(&pb.ChatEvent{Event: &pb.ChatEvent_Restart{Restart: &pb.Restart{
					Reason: "control-plane node lost mid-answer; restarted on another node"}}})
				continue
			}
		}
		if status.Code(err) == codes.Unavailable && attempt < maxRestarts && ctx.Err() == nil {
			continue
		}
		return nil, err
	}
}
