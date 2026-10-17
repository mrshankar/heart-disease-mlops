# Demo video: recording plan and narration (about 8 minutes)

**Before recording (once):** `./run_all.sh` has completed on the VM (all stages `ok`), and you have opened
`reports/evidence_report.html` once to check it renders.

**While recording:** run `./scripts/video_walkthrough.sh` in the VS Code terminal. It shows one step at a time
(title, command, output) and waits for **Enter**. The `Browser:` line tells you which page to switch to.
After a retake, resume with `./scripts/video_walkthrough.sh --from <step>`.

The "Say" text is a guide, not a teleprompter. Learn the points and say them in your own words.
Numbers in **bold** are from the actual run, so quote them exactly.

---

### Step 1: Project overview (0:00-0:40)
**Screen:** terminal: `git log`, repo folders. Optionally the architecture diagram in the README on GitHub.

**Say:**
- "This is my MLOps Assignment 1: a heart disease risk classifier that goes from raw UCI data to a monitored API on Kubernetes."
- "The code is a Python package, `cardiorisk`, with modules for data, features, training, evaluation and serving.
  The repo also holds tests, the Dockerfile, Kubernetes manifests, monitoring configuration and the CI workflow."
- "Everything here runs from this git commit, which is how I keep runs traceable."

### Step 2: Data (0:40-1:30)
**Screen:** terminal: download script, checksums, cleaned CSV.

**Say:**
- "The download script fetches the UCI data and verifies SHA-256 checksums, so the data is versioned and tamper-evident."
- "Instead of only the 303-row Cleveland file, I combined all four hospitals: **918 patients** after removing duplicates."
- "Cleaning turns `?` and impossible zeros into missing values. For example, cholesterol is recorded as 0 for every Swiss patient.
  The 0-4 diagnosis becomes a binary target: about **55% positive**."

### Step 3: Code quality (1:30-2:10)
**Screen:** terminal: flake8 and pytest.

**Say:**
- "flake8 enforces the code style, and **50 tests** cover data cleaning, features, the model, the API contract and the configuration files."
- "One test exists because of a real bug I hit on the Prayogshala VM. `/tmp` is a different filesystem there,
  so moving the plots with a rename failed. The fix uses a copy-and-delete move, and the test simulates that failure so it can't come back."

### Step 4: One-command pipeline (2:10-2:40)
**Screen:** terminal: summary of the last `./run_all.sh` run.

**Say:**
- "`run_all.sh` runs the whole flow: setup, data, quality, train, serve, docker, and optionally Kubernetes.
  Each stage writes its own log, and the script stops at the first failure with a non-zero exit code, like a CI pipeline."
- "On this VM all stages passed, including building and smoke-testing the Docker image."

### Step 5: EDA and model results (2:40-4:00)
**Screen:** terminal table, then `reports/evidence_report.html`: Results, EDA plots, Model evaluation plots.

**Say (EDA):**
- "The positive rate differs a lot between hospitals: **36%** in Hungary versus **93%** in Switzerland. So I stratified the split on hospital and target together."
- "Missing values depend on the hospital too. A 'was missing' flag would mostly tell the model *which hospital* the patient came from.
  So I left those flags out and tested that choice: with them the score was **0.9045**, without them **0.9044**. No gain, so the simpler model stays."
- "I added `hr_reserve_pct`: the max heart rate as a share of the age-predicted maximum (220 minus age)."

**Say (models):**
- "I tuned three model families with 5-fold cross-validation, **72 configurations** in total.
  I selected on cross-validated ROC-AUC only; the test set is used once, at the end."
- "Logistic regression with L1 won: **0.905 ± 0.014**. The tree models overfit more. Their train-to-validation gap is 5 to 6 times larger."
- "For screening, a missed patient is worse than a false alarm. So the threshold maximises F2, which weights recall twice as much,
  with at least 70% specificity. That gives **0.39**. On the test set: ROC-AUC **0.90**, recall **0.91**, specificity **0.73**."

### Step 6: MLflow (4:00-4:50)
**Screen:** http://localhost:5000: run tree (expand it), a model run's metrics and plots, Compare on the grid runs, Models tab.

**Say:**
- "Every training run is tracked in MLflow: one parent run, one run per model family, one per hyper-parameter candidate, plus the ablation and the champion.
  That's **78 runs**, with parameters, metrics, plots and the data hash."
- "Here I compare the random forest grid on a parallel-coordinates plot."
- "The champion is registered as `heart-disease-classifier` with the alias `champion`. The preprocessing is saved inside the model, so there's no training/serving skew."

### Step 7: The API (4:50-5:40)
**Screen:** terminal curls, then http://localhost:8000/docs (Try it out on `POST /predict`).

**Say:**
- "FastAPI serves the model. `/predict` takes JSON and returns the prediction, the probability, the confidence and a risk band."
- "High-risk patient: **0.98**. Low-risk: **0.02**."
- "If optional inputs like `ca` or `thal` are missing, the model imputes them, and the response lists them so the caller knows."
- "Invalid input gets HTTP **422** with a message for each field, so bad data never reaches the model."

### Step 8: Monitoring and logging (5:40-6:30)
**Screen:** http://localhost:8000/monitoring (live), terminal: `/metrics`, JSON logs, alert rules.

**Say:**
- "Every request is logged as one JSON line with a request ID and the latency. The same ID is returned in a response header."
- "`/metrics` exposes Prometheus metrics: request counts, a latency histogram, predictions by risk band, validation errors and missing inputs."
- "There are five alert rules, including *prediction drift*. If the share of positive predictions moves far from the training base rate,
  the incoming patient mix has probably changed."
- "This live dashboard is built into the API. Prometheus and Grafana configs are in the repo for the full stack."

### Step 9: Docker (6:30-7:00)
**Screen:** terminal: image, Dockerfile key lines, container running, prediction from the container.

**Say:**
- "The image is multi-stage, so only the serving dependencies are in the final image.
  It runs as a non-root user and has a health check. The container passes the same smoke tests as the local API."

### Step 10: CI/CD (7:00-7:30)
**Screen:** terminal job list, then GitHub, **Actions** tab, the latest run's graph and summary.

**Say:**
- "GitHub Actions runs five jobs on every push: lint, test, train, docker and deploy-kind.
  Training has a quality gate: test ROC-AUC must be at least **0.85** or the pipeline fails. Every job uploads its logs as artifacts."

### Step 11: Kubernetes (7:30-8:00)
**Screen:** terminal: manifests and key fields.

**Say:**
- "The Deployment runs 2 replicas with readiness and liveness probes, a non-root security context, and rolling updates with no unavailable pods.
  An HPA scales from 2 to 6 pods, and the API is exposed through a LoadBalancer Service and an Ingress."
- "The `preStop` hook gives the load balancer a few seconds to stop sending traffic before a pod shuts down, which avoids errors during rollouts."

### Step 12: Wrap-up (8:00-8:15)
**Say:** "Code, report and evidence are in the GitHub repo. Thank you."
If your course allows AI tools with disclosure, mention here which tools you used and for what.

---

## Likely viva questions: short answers to learn

| Question | Answer in one or two sentences |
|---|---|
| Why combine all four hospitals? | Three times more data and more realistic. The cost is heavy, site-dependent missingness, which the pipeline handles explicitly. |
| Why not accuracy for model selection? | ROC-AUC measures ranking quality across all thresholds. The operating threshold is chosen separately, on recall. |
| Why is the test set used only once? | Choosing models on the test set leaks information and makes the reported score optimistic. |
| How is the threshold chosen? | Maximise F2 on out-of-fold training predictions, with specificity ≥ 0.70. The result is 0.39. |
| What is training/serving skew and how do you avoid it? | Different preprocessing at training and serving time. Here the whole preprocessing is saved inside the sklearn Pipeline artefact. |
| What does MLflow give you here? | Every run's parameters, metrics and plots, a comparison of runs, and a versioned model registry with a `champion` alias. |
| What makes this reproducible? | Pinned requirements, fixed seeds, checksummed data, a data hash in the model version, and the same metrics on a fresh environment. |
| What does the CI quality gate do? | It fails the pipeline if the newly trained model's test ROC-AUC is below 0.85, so a worse model can't be shipped. |
| Readiness vs liveness probe? | Readiness decides whether a pod gets traffic (is the model loaded?). Liveness restarts a pod that has hung. |
| How would you detect drift? | Watch the positive-prediction rate against the 0.55 base rate (there's an alert for it). Next step: per-feature drift such as PSI. |
| Biggest limitation? | Small, old data; weaker performance for hospitals that rarely measure `ca` and `thal`. A screening aid, not a diagnosis. |
