#!/usr/bin/env bash
# =============================================================================
#  demo_walkthrough.sh - guided step-by-step tour of the whole project
#
#  Prerequisite: ./run_all.sh has completed once (model, mlflow.db, logs exist).
#  Run from the repo root:   ./scripts/demo_walkthrough.sh
#  Each step prints a title, shows the command, runs it, then waits for Enter.
#
#  Browser pages open automatically at the right step:
#    - VS Code Remote (laptop -> VM): through VS Code's $BROWSER helper, in your laptop's
#      browser (VS Code forwards the port by itself)
#    - VS Code / terminal inside the VM desktop: through xdg-open, in the VM's browser
#  The evidence report is served on http://localhost:8090 so it opens in both setups.
#
#  Options:  --from N       start at step N (e.g. after a retake)
#            --no-docker    skip the Docker step
#            --no-browser   do not open pages, only print them
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FROM=1; DOCKER=1; OPEN_BROWSER=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift ;;
    --no-docker) DOCKER=0 ;;
    --no-browser) OPEN_BROWSER=0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac; shift
done

PY="${PY:-}"; [[ -z "$PY" && -x .venv/bin/python ]] && PY="$ROOT/.venv/bin/python"; PY="${PY:-python3}"
export PATH="$(dirname "$PY"):$PATH"
B=$'\e[1m'; C=$'\e[36m'; G=$'\e[32m'; Y=$'\e[33m'; N=$'\e[0m'
DEMO_PID=""; REPORT_PID=""
REPORT_PORT="${REPORT_PORT:-8090}"
REPO_URL="$(git config --get remote.origin.url 2>/dev/null | sed -e 's#^git@github.com:#https://github.com/#' -e 's#\.git$##')"
REPO_URL="${REPO_URL:-https://github.com/mrshankar/heart-disease-mlops}"
cleanup() {
  # demo.sh and the report server run in their own sessions (setsid): stop the whole groups
  [[ -n "$DEMO_PID" ]] && { kill -TERM -- "-$DEMO_PID" 2>/dev/null || kill "$DEMO_PID" 2>/dev/null; }
  [[ -n "$REPORT_PID" ]] && { kill -TERM -- "-$REPORT_PID" 2>/dev/null || kill "$REPORT_PID" 2>/dev/null; }
  docker rm -f heart-api-video >/dev/null 2>&1 || true
}
trap cleanup EXIT

STEP=0
step() {  # step "<title>" "<what to show in the browser, optional>"
  STEP=$((STEP + 1))
  [[ $STEP -lt $FROM ]] && return 1
  clear 2>/dev/null || true
  echo "${B}${C}== Step $STEP: $1 ==${N}"
  [[ -n "${2:-}" ]] && echo "${Y}   Browser: $2${N}"
  echo
  return 0
}
run() { echo "${G}\$ $*${N}"; eval "$@"; echo; }
pause() { if { exec 3</dev/tty; } 2>/dev/null; then read -r -p "${B}[Enter] next step${N} " _ <&3; exec 3<&-; else echo; fi; }

latest_logs() { ls -d logs/2*/ 2>/dev/null | sort | tail -1; }

http_ok() { curl -sf -o /dev/null --max-time 3 "$1"; }
wait_http() { for _ in $(seq 1 "${2:-60}"); do http_ok "$1" && return 0; sleep 1; done; return 1; }

open_url() {  # open a page in the presenter's browser (VS Code Remote or VM desktop)
  local url="$1"
  echo "${Y}   -> opening $url${N}"
  [[ $OPEN_BROWSER -eq 1 ]] || return 0
  if [[ -n "${BROWSER:-}" ]]; then                       # VS Code terminal (remote or local)
    "$BROWSER" "$url" >/dev/null 2>&1 &
  elif [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null; then
    xdg-open "$url" >/dev/null 2>&1 &                     # VM desktop
  else
    echo "${Y}   (no browser found from this terminal - open the link above by hand)${N}"
  fi
  sleep 1
}

start_report_server() {  # serve reports/ so the evidence report opens via http:// in any setup
  [[ -f reports/evidence_report.html ]] || "$PY" scripts/evidence_report.py >/dev/null 2>&1
  http_ok "http://localhost:$REPORT_PORT/evidence_report.html" && return 0
  setsid "$PY" -m http.server "$REPORT_PORT" --bind 127.0.0.1 --directory reports > logs/report_server.log 2>&1 < /dev/null &
  REPORT_PID=$!
  wait_http "http://localhost:$REPORT_PORT/evidence_report.html" 15
}

ensure_services() {  # API + MLflow + traffic (also when resuming with --from 7)
  if http_ok "http://localhost:8000/ready" && http_ok "http://localhost:5000"; then return 0; fi
  echo "starting API, MLflow UI and background traffic (demo.sh)..."
  setsid ./demo.sh > logs/video_demo.log 2>&1 < /dev/null &
  DEMO_PID=$!
  wait_http "http://localhost:8000/ready" 60 || { echo "API did not start - see logs/video_demo.log"; tail -5 logs/video_demo.log; }
  wait_http "http://localhost:5000" 60 || echo "MLflow UI did not start - see logs/demo/mlflow.log"
}

# ---------------------------------------------------------------------------
if step "Project overview and version" "GitHub repo page (README with the architecture diagram)"; then
  open_url "$REPO_URL"
  run "git log --oneline -3"
  run "ls"
  run "ls src/cardiorisk src/cardiorisk/serving"
  pause
fi

if step "Data acquisition and checksums"; then
  run "$PY scripts/download_data.py --skip-if-present | tail -3"
  run "(cd data/raw && sha256sum -c CHECKSUMS.sha256)"
  run "head -3 data/processed/heart_clean.csv"
  pause
fi

if step "Code quality: lint + unit tests" ; then
  run "$PY -m flake8 src tests scripts kubeflow && echo 'flake8: no issues'"
  run "$PY -m pytest -q 2>&1 | tail -3"
  pause
fi

if step "One-command pipeline: last full run of ./run_all.sh"; then
  L="$(latest_logs)"
  if [[ -n "$L" && -f "$L/summary.txt" ]]; then run "cat ${L}summary.txt"; else echo "(no summary yet - run ./run_all.sh)"; fi
  run "grep -n 'run_stage ' run_all.sh | head -8"
  pause
fi

if step "Training results and model selection" "reports/evidence_report.html (Results, EDA plots, Model evaluation plots)"; then
  echo "${G}\$ python: model comparison from reports/metrics.json${N}"
  "$PY" - <<'PYEOF'
import json
m = json.load(open("reports/metrics.json")); meta = json.load(open("models/model_metadata.json"))
print(f"{'model':22s} {'CV ROC-AUC':>12s} {'sd':>7s} {'train-CV gap':>13s} {'test ROC-AUC':>13s}")
for k, v in m["models"].items():
    print(f"{k:22s} {v['cv_roc_auc_mean']:12.4f} {v['cv_roc_auc_std']:7.4f} "
          f"{v['overfit_gap_roc_auc']:13.4f} {v['test_roc_auc']:13.4f}")
t = meta["test_metrics"]
print(f"\nchampion={meta['model_family']}  threshold={meta['decision_threshold']}  version={meta['model_version']}")
print(f"test: ROC-AUC {t['test_roc_auc']}  recall {t['test_recall']}  "
      f"specificity {t['test_specificity']}  accuracy {t['test_accuracy']}")
PYEOF
  echo
  start_report_server
  open_url "http://localhost:$REPORT_PORT/evidence_report.html"
  pause
fi

if step "Start the live services (API + MLflow UI + traffic)" "http://localhost:5000  (MLflow: runs, compare, model registry)"; then
  ensure_services
  run "curl -s localhost:8000/ready"
  echo "MLflow: show the run tree, a parallel-coordinates plot, the champion run, the Models tab."
  open_url "http://localhost:5000"
  pause
fi

if step "The prediction API" "http://localhost:8000/docs  (try POST /predict)"; then
  ensure_services
  open_url "http://localhost:8000/docs"
  run "cat samples/high_risk.json"
  run "curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @samples/high_risk.json; echo"
  run "curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @samples/low_risk.json; echo"
  run "curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @samples/partial_input.json; echo"
  run "curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @samples/invalid.json; echo"
  pause
fi

if step "Monitoring and logging" "http://localhost:8000/monitoring  (live dashboard)"; then
  ensure_services
  open_url "http://localhost:8000/monitoring"
  run "curl -s localhost:8000/metrics | grep -E '^cardiorisk_(predictions_total|validation_errors_total)'"
  run "grep '\"msg\": \"request\"' logs/demo/api.log | tail -3 | cut -c1-200"
  run "grep -E 'alert:' monitoring/alert_rules.yml"
  pause
fi

if step "Containerisation (Docker)"; then
  if [[ $DOCKER -eq 1 ]] && command -v docker >/dev/null && timeout 10 docker info >/dev/null 2>&1; then
    run "docker images heart-risk-api"
    run "grep -E '^(FROM|USER|HEALTHCHECK|EXPOSE|CMD)' Dockerfile"
    run "docker run -d --name heart-api-video -p 8010:8000 heart-risk-api:latest >/dev/null && sleep 6 && docker ps --filter name=heart-api-video --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
    run "curl -s -X POST localhost:8010/predict -H 'content-type: application/json' -d @samples/high_risk.json; echo"
    docker rm -f heart-api-video >/dev/null 2>&1
  else
    echo "Docker not available here - show the Dockerfile and the Docker stage output in the evidence report."
    run "grep -E '^(FROM|USER|HEALTHCHECK|EXPOSE|CMD)' Dockerfile"
  fi
  pause
fi

if step "CI/CD (GitHub Actions)" "GitHub -> Actions tab (latest run graph and summary)"; then
  open_url "$REPO_URL/actions"
  run "sed -n '/^jobs:/,\$p' .github/workflows/ci-cd.yml | grep -E '^  [a-z-]+:$'"
  run "grep -n 'MIN_TEST_ROC_AUC' .github/workflows/ci-cd.yml | head -3"
  pause
fi

if step "Kubernetes deployment manifests"; then
  run "ls k8s/base k8s/overlays/k3s k8s/monitoring"
  run "grep -E 'replicas|maxUnavailable|readinessProbe|livenessProbe|preStop|runAsNonRoot' k8s/base/deployment.yaml"
  run "grep -E 'type:|minReplicas|maxReplicas|host:' k8s/base/service.yaml k8s/base/hpa.yaml k8s/base/ingress.yaml"
  if command -v kubectl >/dev/null && timeout 5 kubectl cluster-info >/dev/null 2>&1; then
    run "kubectl get deploy,pods,svc,ingress,hpa -n heart-risk"
  fi
  pause
fi

if step "Wrap-up"; then
  echo "Repository: $REPO_URL"
  echo "Report:     report/MLOps_Assignment1_Report_2025AE05523.docx"
  echo "Evidence:   reports/evidence_report.html"
  echo
  echo "${G}Done - press Enter to stop the demo services.${N}"
  pause
fi
