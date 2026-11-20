# OmniAgent developer entry points.  `make help` lists them.
PY ?= .venv/bin/python
FE := frontend

.PHONY: help install test lint fe-check e2e demo demo-ui export-models helm-lint manifests images cli cli-test security verify

help:  ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## Python venv + deps, frontend deps, Playwright Chromium
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -r requirements-dev.txt -r model_server/requirements-export.txt
	cd $(FE) && npm ci && npx playwright install chromium

test:  ## Backend tests (incl. Helm chart tests when helm is installed)
	$(PY) -m pytest -q

lint:  ## Python + frontend lint
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
	cd $(FE) && npm run lint

fe-check:  ## Frontend: strict typecheck, unit tests, production build
	cd $(FE) && npm run typecheck && npm test && npm run build

e2e:  ## Browser end-to-end tests (boots backend + built UI)
	cd $(FE) && npx playwright test

export-models:  ## Export ONNX models for Triton / the local Triton server
	$(PY) -m model_server.export_onnx

demo:  ## Boot the whole stack, drive concurrent agent users, print a summary
	$(PY) -m scripts.demo_full_stack --users 24

demo-ui:  ## Same, plus the UI, and keep everything running (Ctrl-C to stop)
	$(PY) -m scripts.demo_full_stack --users 12 --ui --serve

helm-lint:  ## Lint the Helm chart (default + production-style values)
	helm lint --strict deploy/kubernetes/helm/omniagent
	helm lint --strict deploy/kubernetes/helm/omniagent --set ingress.tls.enabled=true --set backend.autoscaling.enabled=true

manifests:  ## Regenerate deploy/kubernetes/manifests from the chart
	deploy/kubernetes/render.sh

cli:  ## Build the Go omni CLI into cli/bin/omni
	cd cli && go build -trimpath -ldflags "-X github.com/JCHETAN26/Omni-agent/cli/internal/cmd.Version=$$(git describe --always --dirty)" -o bin/omni ./cmd/omni

cli-test:  ## Go CLI: vet, gofmt, tests
	cd cli && go vet ./... && test -z "$$(gofmt -l .)" && go test ./...

security: cli  ## omni check on this repo (secrets, Dockerfile, Kubernetes, deps)
	cli/bin/omni check --fast .

images:  ## Build container images (needs a running Docker daemon)
	docker build -f deploy/docker/backend.Dockerfile -t omniagent-backend:dev .
	docker build -t omniagent-frontend:dev $(FE)

verify: lint test cli-test security fe-check e2e helm-lint  ## Everything CI would run
