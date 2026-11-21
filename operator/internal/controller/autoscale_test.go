package controller

import (
	"testing"
	"time"
)

func TestDecide(t *testing.T) {
	t0 := time.Unix(1000, 0)
	window := 60 * time.Second
	cases := []struct {
		name     string
		current  int32
		inflight float64
		history  []recommendation
		at       time.Time
		want     int32
	}{
		{"scale up immediately to ceil(inflight/target)", 1, 5, nil, t0, 3},
		{"clamped to max", 1, 100, nil, t0, 4},
		{"never below min", 3, 0, nil, t0, 1},
		{"hold while a recent sample wanted more", 3, 0, []recommendation{{t0.Add(-10 * time.Second), 3}}, t0, 3},
		{"scale down once the window has passed", 3, 0, []recommendation{{t0.Add(-61 * time.Second), 3}}, t0, 1},
		{"partial scale-down to the window's peak", 4, 1, []recommendation{
			{t0.Add(-90 * time.Second), 4}, {t0.Add(-20 * time.Second), 2}}, t0, 2},
	}
	for _, c := range cases {
		got, _ := decide(c.at, c.current, 1, 4, c.inflight, 2, c.history, window)
		if got != c.want {
			t.Errorf("%s: got %d want %d", c.name, got, c.want)
		}
	}
}

func TestDecidePrunesHistory(t *testing.T) {
	t0 := time.Unix(1000, 0)
	hist := []recommendation{{t0.Add(-2 * time.Minute), 4}, {t0.Add(-5 * time.Second), 2}}
	_, kept := decide(t0, 2, 1, 4, 2, 2, hist, time.Minute)
	if len(kept) != 2 || kept[0].replicas != 2 || kept[1].at != t0 {
		t.Fatalf("history %+v", kept)
	}
}
