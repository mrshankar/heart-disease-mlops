#!/usr/bin/env bash
# =============================================================================
#  demo.sh - keep everything running for screenshots and the demo video
#
#  Run ./run_all.sh first (it trains the model and creates mlflow.db). Then:
#      ./demo.sh                    # API :8000 + MLflow UI :5000 + steady traffic
#      ./demo.sh --no-traffic       # without the background load generator
#  Press Ctrl+C to stop everything.
#
#  Open in the VM's browser (or VS Code: Ports tab -> "Open in Browser"):
#      http://localhost:8000/monitoring    live API dashboard (predictions, latency, errors)
#      http://localhost:8000/docs          Swagger UI - try POST /predict
#      http://localhost:8000/metrics       raw Prometheus metrics
#      http://localhost:5000               MLflow UI - runs, metrics, plots, model registry
#      reports/evidence_report.html        every plot, metric and log of the last run
# =============================================================================
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

API_PORT="${API_PORT:-8000}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
TRAFFIC=1
[[ "${1:-}" == "--no-traffic" ]] && TRAFFIC=0

if [[ -x .venv/bin/python ]]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
[[ -f models/model.joblib && -f mlflow.db ]] || { echo "Run ./run_all.sh first (model or mlflow.db missing)."; exit 1; }

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
for p in "$API_PORT" "$MLFLOW_PORT"; do
  if port_busy "$p"; then
    echo "Port $p is already in use (an old API/MLflow still running?)."
    echo "Free it with:  pkill -u \"\$USER\" -f 'uvicorn|mlflow'   or use API_PORT=/MLFLOW_PORT= to pick others."
    exit 1
  fi
done

mkdir -p logs/demo
PIDS=()
STOPPED=""
stop() {
  [[ -n "$STOPPED" ]] && return; STOPPED=1
  echo; echo "stopping demo..."
  # each service runs in its own process group (setsid), so this also stops grandchildren
  # such as the gunicorn workers that `mlflow ui` spawns
  for p in "${PIDS[@]:-}"; do kill -TERM -- "-$p" 2>/dev/null || kill "$p" 2>/dev/null || true; done
  pkill -P $$ 2>/dev/null || true
}
trap stop EXIT INT TERM

setsid "$PY" -m uvicorn cardiorisk.serving.app:app --host 0.0.0.0 --port "$API_PORT" > logs/demo/api.log 2>&1 &
PIDS+=("$!")
setsid "$PY" -m mlflow ui --backend-store-uri "sqlite:///$ROOT/mlflow.db" --host 0.0.0.0 --port "$MLFLOW_PORT" \
  > logs/demo/mlflow.log 2>&1 &
PIDS+=("$!")

for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$API_PORT/ready" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "http://127.0.0.1:$API_PORT/ready" >/dev/null || { echo "API did not start - see logs/demo/api.log"; exit 1; }
for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$MLFLOW_PORT" >/dev/null 2>&1 && break; sleep 1; done

if [[ $TRAFFIC -eq 1 ]]; then
  # gentle, continuous traffic so the dashboards move (about 3 requests/s, 5% invalid)
  setsid bash -c 'while true; do "$0" scripts/load_test.py --url "http://127.0.0.1:$1" --requests 200 \
        --delay 0.3 --seed $RANDOM >/dev/null 2>&1 || true; done' "$PY" "$API_PORT" &
  PIDS+=("$!")
fi

[[ -f reports/evidence_report.html ]] || "$PY" scripts/evidence_report.py >/dev/null 2>&1 || true

cat <<EOF

  Demo is running  (Ctrl+C to stop)
  ----------------------------------------------------------------
  Live API dashboard   http://localhost:$API_PORT/monitoring
  API docs (try it)    http://localhost:$API_PORT/docs
  Prometheus metrics   http://localhost:$API_PORT/metrics
  MLflow UI            http://localhost:$MLFLOW_PORT
  Evidence report      file://$ROOT/reports/evidence_report.html
  ----------------------------------------------------------------
  Sample prediction:
EOF
curl -s -X POST "http://127.0.0.1:$API_PORT/predict" -H 'content-type: application/json' -d @samples/high_risk.json
echo; echo; echo "  JSON request logs (live):"
# stream in the background and `wait`: bash runs traps during `wait`, but would
# defer them until a foreground `tail -f` exits (i.e. never)
setsid bash -c 'tail -n 0 -f logs/demo/api.log | grep --line-buffered "\"msg\": \"prediction\"" | cut -c1-170' &
PIDS+=("$!")
wait
