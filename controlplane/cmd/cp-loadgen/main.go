// cp-loadgen drives closed-loop Search traffic: N workers, each sending its
// next call as soon as the previous one returns, for a fixed duration.
//
//	cp-loadgen --mode grpc   --target dns:///omniagent-controlplane:9000 --workers 8 --duration 120s
//	cp-loadgen --mode direct --direct-url http://omniagent-backend:8000 --workers 8 --duration 60s
//
// grpc goes through the control plane (failover client); direct calls a
// gateway's POST /search with the same body, as the routing-overhead baseline.
// The report has totals, a 100 ms timeline and every call (start offset,
// latency, result, serving node), so fault windows can be analysed afterwards.
// Printed as JSON between REPORT_BEGIN / REPORT_END.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/dns"
	"google.golang.org/grpc/resolver/manual"
	"google.golang.org/grpc/status"

	"github.com/JCHETAN26/Omni-agent/controlplane/client"
	pb "github.com/JCHETAN26/Omni-agent/controlplane/gen/controlv1"
)

var queries = []string{
	"disk error on storage node", "how do I replace a failed drive", "firmware version affected by EIO",
	"rack power and cooling alarm", "network packet loss troubleshooting", "kafka broker certificate rotation",
	"index fund expense ratio", "how are dividends taxed", "should I pay off my mortgage early",
	"what is a good credit score", "how does a 401k match work", "capital gains on stock options",
	"emergency fund size", "difference between roth and traditional ira", "how to read a balance sheet",
	"what happens to my pension if I change jobs", "is renting cheaper than buying", "dollar cost averaging",
	"how do bonds lose value", "tax deduction for home office",
}

type call struct {
	startUs int64 // since the run started
	ms      float64
	code    string
	degr    string
	node    string
}

func pct(v []float64, q float64) float64 {
	if len(v) == 0 {
		return 0
	}
	i := int(q*float64(len(v)-1) + 0.5)
	return float64(int(v[i]*100+0.5)) / 100
}

func main() {
	mode := flag.String("mode", "grpc", "grpc (through the control plane) or direct (gateway HTTP)")
	target := flag.String("target", "dns:///omniagent-controlplane:9000", "control-plane target (grpc mode)")
	directURL := flag.String("direct-url", "http://omniagent-backend:8000", "gateway base URL(s), comma-separated (direct mode)")
	workers := flag.Int("workers", 8, "concurrent closed-loop workers")
	duration := flag.Duration("duration", 2*time.Minute, "")
	timeout := flag.Duration("timeout", 3*time.Second, "per-call deadline")
	label := flag.String("label", "run", "")
	addrs := flag.String("addrs", "", "grpc mode: comma-separated node addresses instead of --target DNS")
	unique := flag.Bool("unique", false, "make every query distinct (defeats the gateway's result cache)")
	flag.Parse()
	log := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	// Replaced control-plane pods get new IPs: re-resolve DNS within a second of
	// a connection failure instead of gRPC's default 30 s minimum.
	dns.SetMinResolutionInterval(time.Second)

	var do func(ctx context.Context, q string) (degr, node string, err error)
	switch *mode {
	case "grpc":
		var opts []grpc.DialOption
		dialTarget := *target
		if *addrs != "" {
			r := manual.NewBuilderWithScheme("static")
			var ra []resolver.Address
			for _, a := range strings.Split(*addrs, ",") {
				ra = append(ra, resolver.Address{Addr: a})
			}
			r.InitialState(resolver.State{Addresses: ra})
			opts, dialTarget = append(opts, grpc.WithResolvers(r)), "static:///nodes"
		}
		conn, router, err := client.Dial(dialTarget, opts...)
		if err != nil {
			log.Error("dial", "err", err)
			os.Exit(1)
		}
		defer conn.Close()
		do = func(ctx context.Context, q string) (string, string, error) {
			r, err := router.Search(ctx, &pb.SearchRequest{Query: q, TopK: 10})
			if err != nil {
				return "", "", err
			}
			d := ""
			if r.Route.Degradation != pb.Degradation_DEGRADATION_NONE {
				d = r.Route.Degradation.String()
			}
			return d, r.Route.Node, nil
		}
	case "direct":
		hc := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 256}}
		urls := strings.Split(*directURL, ",")
		var rr atomic.Int64
		do = func(ctx context.Context, q string) (string, string, error) {
			body, _ := json.Marshal(map[string]any{"query": q, "top_k": 10})
			base := urls[int(rr.Add(1))%len(urls)] // round-robin, like the control plane spreads load
			req, _ := http.NewRequestWithContext(ctx, http.MethodPost, base+"/search", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			resp, err := hc.Do(req)
			if err != nil {
				return "", "", err
			}
			defer resp.Body.Close()
			_, _ = io.Copy(io.Discard, resp.Body)
			if resp.StatusCode != http.StatusOK {
				return "", "", fmt.Errorf("HTTP %d", resp.StatusCode)
			}
			return "", "direct", nil
		}
	default:
		log.Error("unknown mode", "mode", *mode)
		os.Exit(2)
	}

	// Warm up: connections exist (and health checks have run) before timing.
	for range 20 {
		wctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_, _, _ = do(wctx, "warm up")
		cancel()
	}

	var (
		mu    sync.Mutex
		calls []call
		wg    sync.WaitGroup
	)
	start := time.Now()
	end := start.Add(*duration)
	for w := range *workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var local []call
			for i := w; time.Now().Before(end); i += *workers {
				t := time.Now()
				ctx, cancel := context.WithTimeout(context.Background(), *timeout)
				q := queries[i%len(queries)]
				if *unique {
					q = fmt.Sprintf("%s %d", q, i)
				}
				degr, node, err := do(ctx, q)
				cancel()
				c := call{startUs: t.Sub(start).Microseconds(), ms: float64(time.Since(t).Microseconds()) / 1000,
					code: "OK", degr: degr, node: node}
				if err != nil {
					c.code = status.Code(err).String()
					if *mode == "direct" {
						c.code = "ERROR"
					}
				}
				local = append(local, c)
			}
			mu.Lock()
			calls = append(calls, local...)
			mu.Unlock()
		}()
	}
	ticker := time.NewTicker(10 * time.Second)
	go func() {
		for range ticker.C {
			log.Info("progress", "elapsed_s", int(time.Since(start).Seconds()))
		}
	}()
	wg.Wait()
	ticker.Stop()
	sort.Slice(calls, func(i, j int) bool { return calls[i].startUs < calls[j].startUs })

	byCode, nodes := map[string]int{}, map[string]int{}
	degraded := 0
	var lat []float64
	type bucket struct {
		T0Ms     int64   `json:"t_ms"`
		OK       int     `json:"ok"`
		Failed   int     `json:"failed"`
		Degraded int     `json:"degraded"`
		P50      float64 `json:"p50_ms"`
		P99      float64 `json:"p99_ms"`
		lat      []float64
	}
	buckets := map[int64]*bucket{}
	rows := make([][]any, 0, len(calls))
	for _, c := range calls {
		byCode[c.code]++
		k := c.startUs / 100_000
		b := buckets[k]
		if b == nil {
			b = &bucket{T0Ms: k * 100}
			buckets[k] = b
		}
		if c.code != "OK" {
			b.Failed++
		} else {
			b.OK++
			b.lat = append(b.lat, c.ms)
			lat = append(lat, c.ms)
			nodes[c.node]++
			if c.degr != "" {
				b.Degraded++
				degraded++
			}
		}
		rows = append(rows, []any{c.startUs / 1000, c.ms, c.code, c.degr, c.node})
	}
	var timeline []*bucket
	for _, b := range buckets {
		sort.Float64s(b.lat)
		b.P50, b.P99 = pct(b.lat, 0.5), pct(b.lat, 0.99)
		timeline = append(timeline, b)
	}
	sort.Slice(timeline, func(i, j int) bool { return timeline[i].T0Ms < timeline[j].T0Ms })
	sort.Float64s(lat)
	elapsed := time.Since(start).Seconds()
	report := map[string]any{
		"label": *label, "mode": *mode, "workers": *workers, "duration_s": duration.Seconds(), "unique_queries": *unique,
		"timeout_ms": timeout.Milliseconds(), "start_unix_ms": start.UnixMilli(),
		"calls": len(calls), "calls_per_s": float64(int(float64(len(calls))/elapsed*10)) / 10,
		"by_code": byCode, "failed": len(calls) - byCode["OK"], "degraded": degraded,
		"error_rate":   strconv.FormatFloat(float64(len(calls)-byCode["OK"])/float64(max(len(calls), 1)), 'g', 6, 64),
		"latency_ms":   map[string]float64{"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "p99": pct(lat, 0.99), "max": pct(lat, 1)},
		"served_by":    nodes,
		"timeline":     timeline,
		"calls_detail": map[string]any{"columns": []string{"start_ms", "latency_ms", "code", "degradation", "node"}, "rows": rows},
	}
	out, _ := json.Marshal(report)
	fmt.Printf("REPORT_BEGIN\n%s\nREPORT_END\n", out)
}
