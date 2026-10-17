#!/usr/bin/env bash
# =============================================================================
#  demo_walkthrough.sh - guided step-by-step tour of the whole project
#
#  Prerequisite: ./run_all.sh has completed once (model, mlflow.db, logs exist).
#  Run from the repo root:   ./scripts/demo_walkthrough.sh
#  Each step prints a title, shows the command, runs it, then waits for Enter.
#
#  Options:  --from N     start at step N (e.g. after a retake)
#            --no-docker  skip the Docker step
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FROM=1; DOCKER=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift ;;
    --no-docker) DOCKER=0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac; shift
done

PY="${PY:-}"; [[ -z "$PY" && -x .venv/bin/python ]] && PY="$ROOT/.venv/bin/python"; PY="${PY:-python3}"
export PATH="$(dirname "$PY"):$PATH"
B=$'\e[1m'; C=$'\e[36m'; G=$'\e[32m'; Y=$'\e[33m'; N=$'\e[0m'
DEMO_PID=""
cleanup() {
  # demo.sh runs in its own session (setsid), so stop its whole process group
  [[ -n "$DEMO_PID" ]] && { kill -TERM -- "-$DEMO_PID" 2>/dev/null || kill "$DEMO_PID" 2>/dev/null; }
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

# ---------------------------------------------------------------------------
if step "Project overview and version"; then
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
  [[ -f reports/evidence_report.html ]] || $PY scripts/evidence_report.py >/dev/null
  echo "Open reports/evidence_report.html now."
  pause
fi

if step "Start the live services (API + MLflow UI + traffic)" "http://localhost:5000  (MLflow: runs, compare, model registry)"; then
  setsid ./demo.sh > logs/video_demo.log 2>&1 < /dev/null &
  DEMO_PID=$!
  for _ in $(seq 1 60); do curl -sf localhost:8000/ready >/dev/null 2>&1 && break; sleep 1; done
  run "curl -s localhost:8000/ready"
  echo "MLflow UI: http://localhost:5000   - show the run tree, a parallel-coordinates plot, the champion run, Models tab."
  pause
fi

if step "The prediction API" "http://localhost:8000/docs  (try POST /predict)"; then
  run "cat samples/high_risk.json"
  run "curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @samples/high_risk.json; echo"
  run "curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @samples/low_risk.json; echo"
  run "curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @samples/partial_input.json; echo"
  run "curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @samples/invalid.json; echo"
  pause
fi

if step "Monitoring and logging" "http://localhost:8000/monitoring  (live dashboard)"; then
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

if step "CI/CD (GitHub Actions)" "github.com/mrshankar/heart-disease-mlops -> Actions tab"; then
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
  echo "Repository: https://github.com/mrshankar/heart-disease-mlops"
  echo "Report:     report/MLOps_Assignment1_Report_2025AE05523.docx"
  echo "Evidence:   reports/evidence_report.html"
  echo
  echo "${G}Done - press Enter to stop the demo services.${N}"
  pause
fi
