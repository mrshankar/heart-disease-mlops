# Pipeline demo video: recording plan (about 6-7 minutes)

Record the screen with OBS, or with Zoom/Teams by recording a meeting with yourself. Keep each segment short.
Commands assume the repo root.

| # | Time | Show | Say (key points) |
|---|------|------|------------------|
| 1 | 0:00-0:30 | `README.md` architecture diagram | Problem, data (4 hospitals, 918 rows), goal: monitored API on Kubernetes |
| 2 | 0:30-1:15 | `notebooks/01_eda.ipynb` (class balance, missingness by site) | Site-dependent missingness, so stratify on site × target; no missing indicators; engineered `hr_reserve_pct` |
| 3 | 1:15-2:15 | Terminal: `make train`, then `mlflow ui` | Run hierarchy (pipeline > model > 72 grid runs), parallel-coordinates plot, champion run, registry alias `champion` |
| 4 | 2:15-2:45 | Terminal: `make lint test` | 50 tests, coverage, what the tests guard |
| 5 | 2:45-3:30 | GitHub, **Actions** tab, latest run | lint → test → train (quality gate) → docker → deploy-kind; open the job summary and artifacts |
| 6 | 3:30-4:15 | Terminal: `make docker-build docker-run smoke` | Non-root image, healthy status, `/predict` high/low/partial/invalid responses |
| 7 | 4:15-5:30 | Terminal: `kubectl apply -k k8s/base` (or the k3s overlay), `kubectl get all,ingress,hpa -n heart-risk`, `curl` via port-forward or Ingress | Replicas, probes, HPA, Ingress; `kubectl rollout restart` while `scripts/load_test.py` runs (zero failures) |
| 8 | 5:30-6:15 | `docker compose up`, then `scripts/load_test.py --requests 500 --delay 0.05`, Grafana dashboard, Prometheus targets, `/monitoring` | Latency, traffic, drift panel, alert rules, JSON logs (`docker logs heart-api`) |
| 9 | 6:15-6:45 | Prayogshala notebook `00_prayogshala_end_to_end.ipynb` (already executed) | Same pipeline reproduced on the BITS Kubeflow lab |

Tips: increase the terminal font size, run `make train` before recording (it takes about 1 minute), and keep a second terminal running `load_test.py` so the dashboards move.
