# Heart Disease MLOps - common tasks.  `make help`
PY ?= python
IMAGE ?= heart-risk-api
TAG ?= latest

.PHONY: help install data eda train test lint mlflow-ui serve docker-build docker-run smoke \
        compose-up compose-down k8s-deploy k8s-deploy-k3s k8s-monitoring k8s-verify load sync-monitoring kfp all clean

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:         ## clean install from pinned requirements
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

data:            ## download UCI data (falls back to committed raw files) + build clean CSV
	$(PY) scripts/download_data.py --skip-if-present

eda:             ## generate EDA figures into reports/figures
	$(PY) -m cardiorisk.eda

train: data      ## tune 3 model families, log to MLflow, package models/model.joblib
	$(PY) -m cardiorisk.train

lint:            ## flake8
	$(PY) -m flake8 src tests scripts kubeflow

test:            ## unit + integration tests with coverage
	$(PY) -m pytest -v --cov=cardiorisk --cov-report=term-missing

mlflow-ui:       ## browse experiments at http://localhost:5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:           ## run the API locally on :8000
	$(PY) -m uvicorn cardiorisk.serving.app:app --port 8000 --reload

docker-build:    ## build the serving image (needs models/ from `make train`)
	docker build --build-arg GIT_SHA=$$(git rev-parse --short HEAD 2>/dev/null || echo local) \
	  --build-arg BUILD_DATE=$$(date -u +%Y-%m-%dT%H:%M:%SZ) -t $(IMAGE):$(TAG) .

docker-run:      ## run the container on :8000
	docker run -d --rm --name heart-api -p 8000:8000 $(IMAGE):$(TAG)

smoke:           ## call the running API with the sample payloads
	curl -s localhost:8000/ready; echo
	curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @samples/high_risk.json; echo
	curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @samples/low_risk.json; echo
	curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @samples/invalid.json; echo

compose-up:      ## API + Prometheus + Grafana (+ MLflow with --profile tracking)
	docker compose --profile tracking up -d --build

compose-down:
	docker compose --profile tracking down

k8s-deploy:      ## deploy to minikube / kind / Docker Desktop / cloud (ingress-nginx)
	kubectl apply -k k8s/base && kubectl -n heart-risk rollout status deploy/heart-risk-api --timeout=180s

k8s-deploy-k3s:  ## deploy to k3s / Rancher Desktop (Traefik ingress)
	kubectl apply -k k8s/overlays/k3s && kubectl -n heart-risk rollout status deploy/heart-risk-api --timeout=180s

k8s-monitoring:  ## Prometheus + Grafana inside the cluster
	kubectl apply -k k8s/monitoring

k8s-verify:      ## port-forward the Service and hit the endpoints
	kubectl -n heart-risk get all,ingress,hpa,pdb
	kubectl -n heart-risk port-forward svc/heart-risk-api 8080:80 & sleep 3; \
	  curl -s localhost:8080/ready; echo; \
	  curl -s -X POST localhost:8080/predict -H 'content-type: application/json' -d @samples/high_risk.json; echo; \
	  kill %1

load:            ## generate traffic for the dashboards
	$(PY) scripts/load_test.py --url http://localhost:8000 --requests 500 --delay 0.05

sync-monitoring: ## copy dashboard/alerts into the k8s kustomization
	cp monitoring/alert_rules.yml k8s/monitoring/alert_rules.yml
	cp monitoring/grafana/dashboards/heart-risk-api.json k8s/monitoring/heart-risk-api.json

kfp:             ## compile the Kubeflow pipeline (pip install "kfp>=2.7,<3")
	$(PY) kubeflow/pipeline.py --compile-only

all: install lint test train docker-build  ## full local pipeline

clean:
	rm -rf .pytest_cache .coverage htmlcov test-results mlruns
