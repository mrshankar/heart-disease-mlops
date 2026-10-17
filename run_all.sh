#!/usr/bin/env bash
# =============================================================================
#  run_all.sh - execute the entire MLOps flow end to end
#  MLOps Assignment 1 (AIMLCZG523) - Shankaraguru Ramdurg (2025AE05523)
#
#  Stages (in order):
#    setup    create .venv, install pinned requirements + the cardiorisk package
#    data     download UCI data (falls back to committed raw files), verify checksums
#    quality  flake8 lint + pytest (unit, API, config, pipeline smoke tests)
#    train    EDA figures + GridSearchCV training tracked in MLflow + model registry
#    serve    start the FastAPI service, call /predict, batch, invalid input, load test, /metrics
#    docker   build the image, run the container, smoke-test it          (skipped if no docker)
#    k8s      deploy to the current kubectl cluster, rollout, verify      (only with --k8s)
#
#  Usage:
#    ./run_all.sh                       # setup -> serve (+ docker if available)
#    ./run_all.sh --k8s                 # ... and deploy to Kubernetes (kind/minikube/k3s/any)
#    ./run_all.sh --only train          # run one stage (repeatable: --only data --only train)
#    ./run_all.sh --skip docker         # skip a stage (repeatable)
#    ./run_all.sh --no-venv             # use the current python instead of .venv (e.g. Prayogshala notebook kernel)
#    ./run_all.sh --mlflow-ui           # leave the MLflow UI running on :5000 at the end
#
#  Environment overrides: PYTHON=python3.11  API_PORT=8000  IMAGE=heart-risk-api:latest  DOCKER_BUILD_ARGS="--network host"
#  Every stage writes its log to logs/<timestamp>/<stage>.log; the script stops at the
#  first failing stage and prints the tail of that log.
# =============================================================================
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
API_PORT="${API_PORT:-8000}"
IMAGE="${IMAGE:-heart-risk-api:latest}"
USE_VENV=1
DO_K8S=0
KEEP_MLFLOW_UI=0
ONLY=()
SKIP=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --k8s) DO_K8S=1 ;;
    --no-venv) USE_VENV=0 ;;
    --mlflow-ui) KEEP_MLFLOW_UI=1 ;;
    --only) ONLY+=("$2"); shift ;;
    --skip) SKIP+=("$2"); shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

[[ " ${ONLY[*]:-} " == *" k8s "* ]] && DO_K8S=1   # --only k8s implies --k8s

# ----------------------------------------------------------------- helpers
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR"
if [[ -t 1 ]]; then B=$'\e[1m'; G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; N=$'\e[0m'; else B=; G=; R=; Y=; N=; fi
declare -a SUMMARY=()
BG_PIDS=()

cleanup() {
  for p in "${BG_PIDS[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null || true; done
  docker rm -f heart-api-runall >/dev/null 2>&1 || true
}
trap cleanup EXIT

wanted() {   # should stage $1 run?
  local s="$1"
  if [[ ${#ONLY[@]} -gt 0 ]]; then [[ " ${ONLY[*]} " == *" $s "* ]] || return 1; fi
  [[ " ${SKIP[*]:-} " == *" $s "* ]] && return 1
  return 0
}

run_stage() {  # run_stage <name> <function>
  local name="$1" fn="$2" log="$LOG_DIR/$1.log" t0=$SECONDS
  if ! wanted "$name"; then SUMMARY+=("$(printf '%-8s %s' "$name" "${Y}skipped${N}")"); return 0; fi
  echo "${B}==> [$name]${N} (log: logs/$STAMP/$name.log)"
  # NOTE: not "if ( ... )" - bash disables errexit inside an if-condition,
  # which would let a failing command in the stage go unnoticed.
  local rc=0 hb
  # heartbeat: long stages (pip install, tests, grid search) are otherwise silent for minutes
  ( while sleep 30; do
      echo "    ... [$name] still running ($((SECONDS - t0))s) - live log: tail -f logs/$STAMP/$name.log" >&2
    done ) &
  hb=$!
  set +e
  ( set -Eeuo pipefail; "$fn" ) > >(tee "$log") 2>&1
  rc=$?
  set -e
  kill "$hb" 2>/dev/null; wait "$hb" 2>/dev/null || true
  wait 2>/dev/null || true   # let tee flush
  if [[ $rc -eq 0 ]]; then
    SUMMARY+=("$(printf '%-8s %s  (%ss)' "$name" "${G}ok${N}" "$((SECONDS - t0))")")
  else
    SUMMARY+=("$(printf '%-8s %s' "$name" "${R}FAILED${N}")")
    echo; echo "${R}${B}Stage '$name' failed.${N} Last lines of logs/$STAMP/$name.log:"; tail -n 25 "$log"
    print_summary; exit 1
  fi
}

print_summary() {
  echo; echo "${B}================ summary ================${N}"
  printf '  %s\n' "${SUMMARY[@]}"
  echo "  logs: logs/$STAMP/"
}

wait_http() {  # wait_http <url> <seconds>
  local url="$1" n="${2:-60}"
  for _ in $(seq 1 "$n"); do curl -sf "$url" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "timed out waiting for $url"; return 1
}

smoke_api() {  # smoke_api <base-url>
  local base="$1"
  echo "--- GET /ready";  curl -sf "$base/ready"; echo
  for f in high_risk low_risk partial_input; do
    echo "--- POST /predict ($f)"
    curl -sf -X POST "$base/predict" -H 'content-type: application/json' -d @"samples/$f.json"; echo
  done
  echo "--- POST /predict/batch"
  curl -sf -X POST "$base/predict/batch" -H 'content-type: application/json' -d @samples/batch.json | head -c 400; echo
  echo "--- POST /predict (invalid input, expect 422)"
  local code; code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$base/predict" -H 'content-type: application/json' -d @samples/invalid.json)
  echo "HTTP $code"; [[ "$code" == "422" ]]
  "$PY" - "$base" <<'EOF'
import json, sys, urllib.request
base = sys.argv[1]
def post(f):
    req = urllib.request.Request(base + "/predict", data=open(f"samples/{f}.json", "rb").read(),
                                 headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())
hi, lo = post("high_risk"), post("low_risk")
assert hi["prediction"] == 1 and lo["prediction"] == 0, (hi, lo)
print(f"assertions OK: high-risk p={hi['probability']} -> 1, low-risk p={lo['probability']} -> 0")
EOF
}

# ------------------------------------------------------------------ stages
stage_setup() {
  command -v "$PYTHON" >/dev/null || { echo "$PYTHON not found"; return 1; }
  "$PYTHON" - <<'EOF'
import sys
v = sys.version_info
print("python", sys.version.split()[0])
assert (3, 10) <= v[:2] <= (3, 13), "Python 3.10-3.13 required"
EOF
  if [[ $USE_VENV -eq 1 ]]; then
    if [[ ! -x .venv/bin/python ]]; then
      "$PYTHON" -m venv .venv || { echo "venv creation failed - rerun with --no-venv"; return 1; }
    fi
    .venv/bin/python -m pip install -q --upgrade pip
  fi
  echo "installing pinned requirements (first run takes 1-2 minutes)..."
  "$PY" -m pip install --progress-bar off --disable-pip-version-check -r requirements.txt
  "$PY" -m pip install -q -e .
  "$PY" -c "import sklearn, mlflow, fastapi; print('sklearn', sklearn.__version__, '| mlflow', mlflow.__version__, '| fastapi', fastapi.__version__)"
}

stage_data() {
  "$PY" scripts/download_data.py --skip-if-present
  (cd data/raw && sha256sum -c CHECKSUMS.sha256)
}

stage_quality() {
  "$PY" -m flake8 src tests scripts kubeflow && echo "flake8: no issues"
  "$PY" -m pytest -q --cov=cardiorisk --cov-report=term
}

stage_train() {
  export MLFLOW_TRACKING_URI="sqlite:///$ROOT/mlflow.db"
  "$PY" -m cardiorisk.eda > /dev/null && echo "EDA figures written to reports/figures/"
  "$PY" -m cardiorisk.train
  "$PY" - <<'EOF'
import json
m = json.load(open("models/model_metadata.json"))
t = m["test_metrics"]
print(f"\nchampion={m['model_family']}  threshold={m['decision_threshold']}  version={m['model_version']}")
print(f"test ROC-AUC={t['test_roc_auc']}  recall={t['test_recall']}  specificity={t['test_specificity']}  accuracy={t['test_accuracy']}")
assert t["test_roc_auc"] >= 0.85, "quality gate failed: test ROC-AUC < 0.85"
print("quality gate passed (test ROC-AUC >= 0.85)")
EOF
}

stage_serve() {
  [[ -f models/model.joblib ]] || { echo "models/model.joblib missing - run the train stage"; return 1; }
  "$PY" -m uvicorn cardiorisk.serving.app:app --host 0.0.0.0 --port "$API_PORT" > "$LOG_DIR/api.log" 2>&1 &
  local pid=$!; BG_PIDS+=("$pid")
  wait_http "http://127.0.0.1:$API_PORT/ready" 60
  smoke_api "http://127.0.0.1:$API_PORT"
  echo "--- load test (300 requests)"
  "$PY" scripts/load_test.py --url "http://127.0.0.1:$API_PORT" --requests 300
  echo "--- /metrics excerpt"
  curl -sf "http://127.0.0.1:$API_PORT/metrics" | grep -E '^cardiorisk_predictions_total' || true
  echo "--- last request log lines"
  tail -n 3 "$LOG_DIR/api.log"
  kill "$pid" 2>/dev/null || true
}

stage_docker() {
  # `docker info` can hang forever when the daemon socket exists but the daemon is stuck
  if ! command -v docker >/dev/null || ! timeout 15 docker info >/dev/null 2>&1; then
    echo "docker not available or not responding here - skipping (CI builds and tests the image)"; return 0
  fi
  # DOCKER_BUILD_ARGS lets you add e.g. "--network host" or "--build-arg BASE_IMAGE=..."
  # shellcheck disable=SC2086
  timeout "${DOCKER_BUILD_TIMEOUT:-900}" docker build ${DOCKER_BUILD_ARGS:-} --build-arg GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo local)" \
               --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" -t "$IMAGE" .
  docker rm -f heart-api-runall >/dev/null 2>&1 || true
  local port=$((API_PORT + 1))
  docker run -d --name heart-api-runall -p "$port:8000" "$IMAGE"
  wait_http "http://127.0.0.1:$port/ready" 60
  smoke_api "http://127.0.0.1:$port"
  sleep 16; echo "container health: $(docker inspect --format '{{.State.Health.Status}}' heart-api-runall)"
  docker logs heart-api-runall 2>&1 | tail -n 3
  docker rm -f heart-api-runall >/dev/null
}

stage_k8s() {
  command -v kubectl >/dev/null || { echo "kubectl not installed"; return 1; }
  kubectl cluster-info >/dev/null || { echo "no reachable cluster in the current kubectl context"; return 1; }
  local ctx; ctx="$(kubectl config current-context 2>/dev/null || echo unknown)"
  echo "kubectl context: $ctx"
  # make the locally built image visible to the cluster
  if [[ "$ctx" == kind-* ]]; then kind load docker-image "$IMAGE" --name "${ctx#kind-}"
  elif [[ "$ctx" == minikube ]]; then minikube image load "$IMAGE"
  else echo "assuming the cluster can see image $IMAGE (Docker Desktop / k3s --docker / registry)"; fi
  local overlay="k8s/base"
  kubectl get ingressclass traefik >/dev/null 2>&1 && overlay="k8s/overlays/k3s"
  echo "applying $overlay"
  kubectl apply -k "$overlay"
  kubectl -n heart-risk set image deployment/heart-risk-api api="$IMAGE"
  kubectl -n heart-risk rollout status deployment/heart-risk-api --timeout=180s
  # rollout status returns while old pods are still terminating; a port-forward
  # bound to one of those would die mid-test, so wait until only Running pods remain
  for _ in $(seq 1 60); do
    [[ "$(kubectl -n heart-risk get pods -l app=heart-risk-api --no-headers | grep -vc ' Running ')" == "0" ]] && break
    sleep 2
  done
  kubectl -n heart-risk get deploy,pods,svc,ingress,hpa,pdb -o wide
  # reach the Service: kubectl port-forward first, NodePort as a fallback
  # (port-forward needs socat on some nodes, e.g. k3s with the docker runtime)
  local pf_port=$((API_PORT + 2)) url="" pf_pid
  kubectl -n heart-risk port-forward svc/heart-risk-api "$pf_port:80" > "$LOG_DIR/port-forward.log" 2>&1 &
  pf_pid=$!; BG_PIDS+=("$pf_pid")
  if wait_http "http://127.0.0.1:$pf_port/ready" 15 && curl -sf "http://127.0.0.1:$pf_port/health" >/dev/null; then
    url="http://127.0.0.1:$pf_port"; echo "using kubectl port-forward: $url"
  else
    kill "$pf_pid" 2>/dev/null || true
    local node_ip node_port
    node_ip="$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')"
    node_port="$(kubectl -n heart-risk get svc heart-risk-api -o jsonpath='{.spec.ports[0].nodePort}')"
    url="http://$node_ip:$node_port"; echo "port-forward unavailable, using Service NodePort: $url"
    wait_http "$url/ready" 30
  fi
  smoke_api "$url"
  "$PY" scripts/load_test.py --url "$url" --requests 200
  kill "$pf_pid" 2>/dev/null || true
  echo "Monitoring (optional): kubectl apply -k k8s/monitoring  -> Prometheus :30090, Grafana :30300 (NodePorts)"
}

# -------------------------------------------------------------------- main
if [[ $USE_VENV -eq 1 ]]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v "$PYTHON")"; fi
export PY
echo "${B}Heart Disease MLOps - full pipeline run ($STAMP)${N}"
echo "repo: $ROOT | python: $PY"

run_stage setup   stage_setup
run_stage data    stage_data
run_stage quality stage_quality
run_stage train   stage_train
run_stage serve   stage_serve
run_stage docker  stage_docker
if [[ $DO_K8S -eq 1 ]]; then run_stage k8s stage_k8s; else SUMMARY+=("$(printf '%-8s %s' k8s "${Y}skipped (use --k8s)${N}")"); fi

if [[ $KEEP_MLFLOW_UI -eq 1 ]]; then
  nohup "$PY" -m mlflow ui --backend-store-uri "sqlite:///$ROOT/mlflow.db" --host 0.0.0.0 --port 5000 \
    > "$LOG_DIR/mlflow-ui.log" 2>&1 &
  disown || true
  echo "MLflow UI running on http://localhost:5000 (pid $!)"
fi

print_summary
echo "${G}${B}All requested stages completed.${N}"
