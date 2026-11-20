# omni — OmniAgent developer CLI

A Go ([Cobra](https://github.com/spf13/cobra)) CLI that generates new services,
checks code before review, and talks to a running OmniAgent gateway.

```bash
make cli                     # builds cli/bin/omni (from the repo root)
go install ./cmd/omni        # or put it on your PATH (from cli/)
```

## Generate a service

```bash
omni new service incident-notes --lang go          # stdlib Go HTTP service
omni new service ticket-router --lang python       # FastAPI service
omni new service ticket-router --lang python --verify   # also run its tests
```

Each service gets health and readiness endpoints, graceful SIGTERM draining
(readiness fails first, then in-flight requests finish), Prometheus metrics,
tests, a Makefile, a non-root Dockerfile with a `HEALTHCHECK`, and a Kubernetes
manifest with probes, resource limits, a read-only root filesystem, dropped
capabilities and `RuntimeDefault` seccomp. Python dependencies are pinned with
`==` so audits match what ships.

After generating, `omni new` runs `omni check` on the result. The generated
code is held to zero findings by `internal/scaffold` tests. The one exception
is `deps/go-toolchain`, which reports on the Go installed on your machine.

## Check code before review

```bash
omni check                   # current directory
omni check --fail-on medium  # exit 1 at medium or above (default: high)
omni check --fast            # built-in rules only, no external tools
omni check --json            # machine-readable report
omni hooks install           # pre-commit hook: omni check --staged
```

| Area | Rules |
| --- | --- |
| Secrets | private keys; AWS, GitHub, OpenAI, Anthropic, Slack, Google keys; high-entropy values assigned to credential-named settings; `.env` files with values |
| Dockerfile (final stage) | runs as root, unpinned or `:latest` base, `curl \| sh`, remote `ADD`, `sudo`, credentials in `ENV`/`ARG`, missing `HEALTHCHECK` |
| Kubernetes (rendered YAML) | privileged, privilege escalation, root UID, writable root filesystem, kept or added capabilities, host namespaces, `hostPath`, missing memory limit or CPU request, unpinned image, missing readiness probe |
| Python dependencies | unpinned packages, plain-HTTP indexes, `--extra-index-url` (dependency confusion) |
| External tools | `go vet`, `gofmt`, `govulncheck` (reachable vs. merely imported; standard-library advisories rolled into one toolchain upgrade), `ruff`, `pip-audit` |

External tools run when installed. The report lists any that were skipped.
Inside a git work tree, files ignored by `.gitignore` are not scanned. `--staged`
scans the index versions of staged files, so the hook checks exactly what is
being committed. Mark a known-safe line with `omni:allow-secret`.

## Query a gateway

```bash
omni connect --server http://127.0.0.1:8000
omni sources add /path/to/runbooks       # upserts nonempty Markdown files
omni ask "What does EIO-5 mean on storage-07?"
omni ask --continue "Which one should I use?"
omni ask --json "..."                    # outcome, sources, thread id, TTFT and total ms
```

`ask` streams `/sse/agent` and rejects out-of-order events or tokens, and any
answer whose token count or SHA-256 differs from the gateway's `response.end`.
State lives in `$XDG_CONFIG_HOME/omniagent/cli.json`.
