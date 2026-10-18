#!/usr/bin/env bash
# =============================================================================
#  capture_figures.sh - re-create the report's terminal figures from THIS machine
#
#    Fig 11  flake8 + pytest with coverage            -> screenshots/01_tests_terminal.png
#    Fig 12  Docker build, healthy container, samples -> screenshots/09_docker_terminal.png
#    Fig 13  Kubernetes resources + rollout history   -> screenshots/17_k8s_terminal.png
#    Fig 14  Ingress/Service checks, load test,
#            zero-downtime rolling restart            -> screenshots/18_k8s_verify_terminal.png
#
#  Usage (repo root, after ./run_all.sh has trained the model):
#    ./scripts/capture_figures.sh               # all four figures
#    ./scripts/capture_figures.sh --only 11     # one figure (repeatable: --only 13 --only 14)
#    ./scripts/capture_figures.sh --cleanup     # delete the kind cluster this script created
#
#  Kubernetes (figs 13-14): uses the current kubectl cluster if one is reachable;
#  otherwise creates a local kind cluster "mlops-figs" with ingress-nginx
#  (downloads kind + kubectl into ~/.local/bin, no sudo; needs Docker + internet).
#  Raw text of every figure is kept in screenshots/raw/ as well.
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ONLY=(); CLEANUP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY+=("$2"); shift ;;
    --cleanup) CLEANUP=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac; shift
done
want() { [[ ${#ONLY[@]} -eq 0 || " ${ONLY[*]} " == *" $1 "* ]]; }

PY="${PY:-}"; [[ -z "$PY" && -x .venv/bin/python ]] && PY="$ROOT/.venv/bin/python"; PY="${PY:-python3}"
export PATH="$HOME/.local/bin:$(dirname "$PY"):$PATH"
IMAGE="heart-risk-api:latest"
KIND_NAME="mlops-figs"
KIND_INGRESS_PORT="${KIND_INGRESS_PORT:-8088}"
OUT="screenshots"; RAW="$OUT/raw"; mkdir -p "$RAW"
B=$'\e[1m'; G=$'\e[32m'; R=$'\e[31m'; N=$'\e[0m'
BG_PIDS=()
cleanup_bg() { for p in "${BG_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; docker rm -f heart-api-fig >/dev/null 2>&1 || true; }
trap cleanup_bg EXIT

if [[ $CLEANUP -eq 1 ]]; then
  command -v kind >/dev/null && kind delete cluster --name "$KIND_NAME"; exit 0
fi

# ------------------------------------------------------------------ helpers
cap() {  # cap <file> <command...>  -> prints "$ command", runs it, appends both to file and screen
  local f="$1"; shift
  { echo "\$ $*"; bash -c "$*" 2>&1; echo; } | tee -a "$f"
}
render() {  # render <rawfile> <png> <title>
  "$PY" - "$1" "$2" "$3" "$(hostname)" <<'PYEOF'
import re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
raw, png, title, host = sys.argv[1:5]
text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", open(raw, errors="replace").read()).rstrip("\n")
lines = []
for ln in text.splitlines():
    ln = ln.replace("\t", "    ")
    while len(ln) > 175:
        lines.append(ln[:175]); ln = "    " + ln[175:]
    lines.append(ln)
lines = lines[:150]
lh, w = 0.155, 15.5
h = 0.55 + lh * (len(lines) + 1)
fig = plt.figure(figsize=(w, h), dpi=110)
fig.patch.set_facecolor("#0e1116")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, w); ax.set_ylim(h, 0)
ax.add_patch(plt.Rectangle((0, 0), w, 0.36, color="#1c2129"))
for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
    ax.add_patch(plt.Circle((0.22 + i * 0.22, 0.18), 0.065, color=c))
ax.text(0.95, 0.19, f"{host}: heart-disease-mlops - {title}", color="#9aa5b1", fontsize=8.5, va="center", family="DejaVu Sans")
y = 0.55
for ln in lines:
    color, weight = "#d6dde6", "normal"
    if ln.startswith("$ "): color, weight = "#7ee787", "bold"
    elif ln.startswith("#"): color = "#8b949e"
    elif re.search(r"\bPASSED\b|successfully|healthy|\b(passed|OK)\b", ln): color = "#a5e8b0"
    elif re.search(r"FAILED|Error|error:|Traceback", ln): color = "#ff7b72"
    ax.text(0.15, y, ln, color=color, fontsize=7.6, family="DejaVu Sans Mono", va="top", fontweight=weight)
    y += lh
fig.savefig(png, facecolor=fig.get_facecolor())
print("saved", png)
PYEOF
}
banner() { echo; echo "${B}==== $* ====${N}"; }

# ---------------------------------------------------------------- Figure 11
if want 11; then
  banner "Figure 11: lint + tests"
  F="$RAW/01_tests.txt"; : > "$F"
  cap "$F" "flake8 src tests scripts kubeflow && echo 'flake8: no issues'"
  { echo "\$ pytest -v --cov=cardiorisk"
    "$PY" -m pytest -v --cov=cardiorisk --cov-report=term -p no:cacheprovider 2>&1 \
      | grep -E "PASSED|FAILED|ERROR|passed|failed|^TOTAL" | sed 's/ *\[ *[0-9]*%\]//'
  } | tee -a "$F"
  render "$F" "$OUT/01_tests_terminal.png" "flake8 + pytest (clean venv, pinned requirements)"
fi

# ---------------------------------------------------------------- Figure 12
if want 12; then
  banner "Figure 12: Docker"
  F="$RAW/09_docker.txt"; : > "$F"
  if ! command -v docker >/dev/null || ! timeout 15 docker info >/dev/null 2>&1; then
    echo "${R}Docker is not available - skipping figure 12${N}"
  else
    echo "building $IMAGE (cached layers make this quick)..."
    docker build -q --build-arg GIT_SHA="$(git rev-parse --short HEAD)" --build-arg BUILD_DATE="$(date -u +%FT%TZ)" -t "$IMAGE" . >/dev/null \
      || { echo "${R}docker build failed${N}"; exit 1; }
    echo "\$ docker build -t $IMAGE .   (multi-stage, non-root, HEALTHCHECK)" >> "$F"
    cap "$F" "docker images heart-risk-api --format 'table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedSince}}\t{{.Size}}'"
    docker rm -f heart-api-fig >/dev/null 2>&1
    cap "$F" "docker run -d --name heart-api-fig -p 8020:8000 $IMAGE | cut -c1-12"
    for _ in $(seq 1 40); do [[ "$(docker inspect --format '{{.State.Health.Status}}' heart-api-fig 2>/dev/null)" == healthy ]] && break; sleep 1; done
    cap "$F" "docker ps --filter name=heart-api-fig --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'"
    for s in high_risk low_risk partial_input invalid; do
      cap "$F" "curl -s -X POST localhost:8020/predict -H 'content-type: application/json' -d @samples/$s.json; echo"
    done
    cap "$F" "docker logs heart-api-fig 2>&1 | tail -3 | cut -c1-200"
    docker rm -f heart-api-fig >/dev/null 2>&1
    render "$F" "$OUT/09_docker_terminal.png" "Docker image build & container smoke test"
  fi
fi

# --------------------------------------------------------- Kubernetes setup
k8s_ready() { command -v kubectl >/dev/null && timeout 10 kubectl cluster-info >/dev/null 2>&1; }
install_tools() {
  mkdir -p "$HOME/.local/bin"
  if ! command -v kind >/dev/null; then
    echo "installing kind into ~/.local/bin ..."
    curl -fsSL -o "$HOME/.local/bin/kind" https://github.com/kubernetes-sigs/kind/releases/download/v0.29.0/kind-linux-amd64 && chmod +x "$HOME/.local/bin/kind"
  fi
  if ! command -v kubectl >/dev/null; then
    echo "installing kubectl into ~/.local/bin ..."
    curl -fsSL -o "$HOME/.local/bin/kubectl" https://dl.k8s.io/release/v1.33.1/bin/linux/amd64/kubectl && chmod +x "$HOME/.local/bin/kubectl"
  fi
  command -v kind >/dev/null && command -v kubectl >/dev/null
}
INGRESS_URL=""; USING_KIND=0
setup_k8s() {
  if k8s_ready; then
    echo "using existing cluster: $(kubectl config current-context)"
  else
    command -v docker >/dev/null || { echo "${R}No cluster and no Docker - cannot create one${N}"; return 1; }
    install_tools || { echo "${R}could not install kind/kubectl (no internet?)${N}"; return 1; }
    if ! kind get clusters 2>/dev/null | grep -qx "$KIND_NAME"; then
      echo "creating kind cluster '$KIND_NAME' (first time: 2-4 minutes)..."
      cat > /tmp/kind-figs.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      - {containerPort: 80, hostPort: $KIND_INGRESS_PORT, protocol: TCP}
EOF
      kind create cluster --name "$KIND_NAME" --config /tmp/kind-figs.yaml --wait 120s || return 1
    fi
    kubectl config use-context "kind-$KIND_NAME" >/dev/null
    if ! kubectl get ns ingress-nginx >/dev/null 2>&1; then
      echo "installing ingress-nginx..."
      kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.1/deploy/static/provider/kind/deploy.yaml >/dev/null
    fi
    kubectl -n ingress-nginx wait --for=condition=ready pod -l app.kubernetes.io/component=controller --timeout=240s >/dev/null || return 1
    USING_KIND=1
  fi
  # image into the cluster
  docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -q -t "$IMAGE" . >/dev/null
  local ctx; ctx="$(kubectl config current-context)"
  if [[ "$ctx" == kind-* ]]; then kind load docker-image "$IMAGE" --name "${ctx#kind-}" >/dev/null
  elif [[ "$ctx" == minikube ]]; then minikube image load "$IMAGE"; fi
  # deploy (traefik clusters such as k3s use the overlay)
  local overlay="k8s/base"; kubectl get ingressclass traefik >/dev/null 2>&1 && overlay="k8s/overlays/k3s"
  kubectl apply -k "$overlay" >/dev/null
  kubectl -n heart-risk annotate deployment/heart-risk-api kubernetes.io/change-cause="initial deploy of $IMAGE" --overwrite >/dev/null
  kubectl -n heart-risk rollout status deployment/heart-risk-api --timeout=180s >/dev/null || return 1
  # second revision so the rollout history has something to show
  kubectl -n heart-risk rollout restart deployment/heart-risk-api >/dev/null
  kubectl -n heart-risk annotate deployment/heart-risk-api kubernetes.io/change-cause="rolling restart (zero-downtime check)" --overwrite >/dev/null
  kubectl -n heart-risk rollout status deployment/heart-risk-api --timeout=180s >/dev/null
  for _ in $(seq 1 60); do [[ "$(kubectl -n heart-risk get pods -l app=heart-risk-api --no-headers | grep -vc ' Running ')" == 0 ]] && break; sleep 2; done
  # ingress address
  if [[ $USING_KIND -eq 1 ]]; then INGRESS_URL="http://localhost:$KIND_INGRESS_PORT"
  else
    local ip; for _ in $(seq 1 30); do ip="$(kubectl -n heart-risk get ingress heart-risk-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)"; [[ -n "$ip" ]] && break; sleep 2; done
    INGRESS_URL="http://${ip:-127.0.0.1}"
  fi
}

if want 13 || want 14; then
  banner "Kubernetes setup"
  if ! setup_k8s; then
    echo "${R}Kubernetes not available - figures 13 and 14 skipped.${N}"
    ONLY=(none)
  fi
fi

# ---------------------------------------------------------------- Figure 13
if want 13; then
  banner "Figure 13: Kubernetes resources"
  F="$RAW/17_k8s.txt"; : > "$F"
  cap "$F" "kubectl get nodes -o wide"
  cap "$F" "kubectl get deploy,pods,svc,ingress,hpa,pdb -n heart-risk -o wide | cut -c1-200"
  cap "$F" "kubectl -n heart-risk rollout history deployment/heart-risk-api"
  render "$F" "$OUT/17_k8s_terminal.png" "Kubernetes deployment state"
fi

# ---------------------------------------------------------------- Figure 14
if want 14; then
  banner "Figure 14: endpoint checks + zero-downtime rollout"
  F="$RAW/18_k8s_verify.txt"; : > "$F"
  H="Host: heart-risk.local"
  cap "$F" "curl -s -H '$H' $INGRESS_URL/ready; echo        # through the Ingress"
  cap "$F" "curl -s -H '$H' -X POST $INGRESS_URL/predict -H 'content-type: application/json' -d @samples/high_risk.json; echo"
  # Service: port-forward, NodePort fallback
  SVC_URL=""
  kubectl -n heart-risk port-forward svc/heart-risk-api 8030:80 >/tmp/pf-figs.log 2>&1 &
  BG_PIDS+=("$!")
  for _ in $(seq 1 15); do curl -sf localhost:8030/health >/dev/null 2>&1 && SVC_URL="http://localhost:8030" && break; sleep 1; done
  if [[ -z "$SVC_URL" ]]; then
    np="$(kubectl -n heart-risk get svc heart-risk-api -o jsonpath='{.spec.ports[0].nodePort}')"
    nip="$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')"
    SVC_URL="http://$nip:$np"
  fi
  cap "$F" "\"$PY\" scripts/load_test.py --url $SVC_URL --requests 300 --seed 21        # through the Service"
  echo "# zero-downtime check: continuous traffic through the Ingress during 'kubectl rollout restart'" | tee -a "$F"
  "$PY" - "$INGRESS_URL" > /tmp/zd-figs.txt 2>&1 <<'PYEOF' &
import sys, time, urllib.request, urllib.error
url = sys.argv[1] + "/predict"; body = open("samples/high_risk.json", "rb").read()
codes, end = {}, time.time() + 35
while time.time() < end:
    req = urllib.request.Request(url, data=body, headers={"Host": "heart-risk.local", "content-type": "application/json"})
    try: c = urllib.request.urlopen(req, timeout=3).status
    except urllib.error.HTTPError as e: c = e.code
    except Exception as e: c = type(e).__name__
    codes[c] = codes.get(c, 0) + 1; time.sleep(0.02)
print("ingress traffic during rollout:", codes)
PYEOF
  ZD=$!
  sleep 2
  cap "$F" "kubectl -n heart-risk rollout restart deployment/heart-risk-api"
  cap "$F" "kubectl -n heart-risk rollout status deployment/heart-risk-api --timeout=180s | tail -1"
  wait "$ZD"; tee -a "$F" < /tmp/zd-figs.txt
  render "$F" "$OUT/18_k8s_verify_terminal.png" "Endpoint verification via Ingress / Service + zero-downtime rollout"
fi

echo
echo "${G}${B}Done.${N} Figures are in $OUT/ (raw text in $RAW/)."
if [[ $USING_KIND -eq 1 ]]; then echo "The kind cluster '$KIND_NAME' is still running; remove it with: ./scripts/capture_figures.sh --cleanup"; fi
exit 0
