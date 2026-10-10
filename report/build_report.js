// Builds report/MLOps_Assignment1_Report_2025AE05523.docx from project outputs.
// node report/build_report.js   (run from the repo root)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  HeadingLevel, WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, Header,
  PageNumber, TableOfContents, ExternalHyperlink,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const M = JSON.parse(fs.readFileSync(path.join(ROOT, "reports/metrics.json")));
const META = JSON.parse(fs.readFileSync(path.join(ROOT, "models/model_metadata.json")));
const EDA = JSON.parse(fs.readFileSync(path.join(ROOT, "reports/eda_summary.json")));
const REPO_URL = process.env.REPO_URL || "https://github.com/<your-github-username>/heart-disease-mlops";

const FONT = "Calibri";
const W = 9638; // A4 text width in DXA with 2 cm margins
const f = (x, d = 3) => Number(x).toFixed(d);

function pngSize(p) {
  const b = fs.readFileSync(p);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function img(rel, widthIn = 6.3, caption) {
  const p = path.join(ROOT, rel);
  const { w, h } = pngSize(p);
  const wp = Math.round(widthIn * 96);
  const out = [new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(p), transformation: { width: wp, height: Math.round(wp * h / w) } })],
  })];
  if (caption) out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: caption, italics: true, size: 18, color: "555555" })] }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ spacing: { after: 100, line: 276 }, ...opts,
  children: (Array.isArray(text) ? text : [text]).map(t => typeof t === "string" ? new TextRun(t) : t) });
const B = (t) => new TextRun({ text: t, bold: true });
const C = (t) => new TextRun({ text: t, font: "Consolas", size: 19 });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (parts, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, spacing: { after: 60 },
  children: (Array.isArray(parts) ? parts : [parts]).map(t => typeof t === "string" ? new TextRun(t) : t) });
const code = (lines) => lines.map((l, i) => new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: "F3F4F6", color: "auto" },
  spacing: { before: i === 0 ? 60 : 0, after: i === lines.length - 1 ? 120 : 0 },
  indent: { left: 200, right: 200 },
  children: [new TextRun({ text: l || " ", font: "Consolas", size: 17 })] }));

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFC5CD" };
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head, shade) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: head ? { type: ShadingType.CLEAR, fill: "1F3A5F", color: "auto" } : shade ? { type: ShadingType.CLEAR, fill: "F3F6FA", color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ children: [new TextRun({ text: String(t), bold: head, color: head ? "FFFFFF" : undefined, size: 18 })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((t, i) => cell(t, i, false, ri % 2 === 1)) }))],
  });
}
const gap = () => new Paragraph({ spacing: { after: 80 }, children: [] });
function note(text) {
  return new Paragraph({ shading: { type: ShadingType.CLEAR, fill: "FFF7E6", color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: "E0A030", space: 8 } },
    spacing: { before: 80, after: 140 }, indent: { left: 120 },
    children: [new TextRun({ text, size: 19 })] });
}

const models = M.models;
const order = ["logistic_regression", "random_forest", "gradient_boosting"];
const nice = { logistic_regression: "Logistic Regression (L1)", random_forest: "Random Forest", gradient_boosting: "HistGradientBoosting" };
const T = META.test_metrics;

const children = [
  // ------------------------------------------------------------ title page
  new Paragraph({ spacing: { before: 1800, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Machine Learning Operations (AIMLCZG523)", size: 26, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "Assignment 01", size: 26, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400, after: 200 },
    children: [new TextRun({ text: "Heart Disease Risk Prediction", bold: true, size: 52, color: "1F3A5F" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 800 },
    children: [new TextRun({ text: "End-to-end ML model development, CI/CD and production deployment", size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Shankaraguru Ramdurg", bold: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "BITS ID 2025AE05523", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "M.Tech AI & ML (WILP), BITS Pilani", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400, after: 80 }, children: [
    new TextRun({ text: "Code repository: ", size: 22 }),
    new ExternalHyperlink({ link: REPO_URL, children: [new TextRun({ text: REPO_URL, style: "Hyperlink", size: 22 })] })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "October 2026", size: 22, color: "555555" })] }),
  new Paragraph({ children: [new PageBreak()] }),


  // ------------------------------------------------------------ 1 summary
  H1("1. Summary"),
  P(["This project builds a classifier that predicts the presence of heart disease from 13 routine clinical measurements, and runs it as a monitored, containerised REST API on Kubernetes. Every stage is automated and reproducible: data acquisition with checksums, cleaning, EDA, tuned model training with full MLflow tracking and a model registry, 50 automated tests, a five-stage GitHub Actions pipeline, a hardened Docker image, Kubernetes manifests with autoscaling and zero-downtime rollouts, and Prometheus/Grafana monitoring. The same code also runs end to end inside BITS Prayogshala (Kubeflow) from a single notebook or as a Kubeflow Pipeline."]),
  table(["Item", "Result"], [
    ["Data", "All four UCI hospitals combined: 920 rows, 918 after de-duplication, 55.3% positive"],
    ["Champion model", `L1 logistic regression (C = ${META.best_params.C}) in a single sklearn Pipeline`],
    ["5-fold CV ROC-AUC", `${f(META.cv_metrics.cv_roc_auc_mean)} ± ${f(META.cv_metrics.cv_roc_auc_std)}`],
    ["Hold-out test (184 patients)", `ROC-AUC ${f(T.test_roc_auc)}, recall ${f(T.test_recall)}, specificity ${f(T.test_specificity)}, accuracy ${f(T.test_accuracy)} at threshold ${META.decision_threshold}`],
    ["Experiment tracking", "78 MLflow runs (pipeline, 3 model families, 72 grid candidates, ablation, champion), registered model heart-disease-classifier@champion"],
    ["Quality", "50 pytest tests, 97% line coverage, flake8 clean; clean-environment install verified"],
    ["Deployment", "Docker image (non-root, health check) on Kubernetes: Deployment (2-6 pods, HPA), LoadBalancer Service, Ingress, PDB"],
    ["Monitoring", "Prometheus metrics with 5 alert rules, Grafana dashboard (12 panels), JSON request logs, /monitoring page"],
  ], [2600, 7038]),

  // ------------------------------------------------------------ 2 architecture
  H1("2. Architecture"),
  P("The system has three layers. The experimentation layer turns raw UCI files into a versioned, packaged model. The serving layer runs that model behind a validated API on Kubernetes, where it is scraped by Prometheus. The automation layer (GitHub Actions, plus Prayogshala/Kubeflow) re-runs everything on each change and blocks releases that fail a gate."),
  ...img("reports/figures/architecture.png", 6.5, "Figure 1 - System architecture"),
  P([B("Key design principle: one artefact, one code path. "), "Feature engineering, imputation, scaling and encoding live inside the saved sklearn Pipeline (models/model.joblib). The API, the MLflow pyfunc model, the notebooks and the Kubeflow pipeline all import the same src/cardiorisk package, so there is no training/serving skew."]),

  // ------------------------------------------------------------ 3 data & EDA
  H1("3. Data acquisition and EDA"),
  H2("3.1 Acquisition"),
  P(["scripts/download_data.py downloads the UCI archive (dataset 45), falling back to the legacy per-file URLs or to a local zip (--zip) on machines without internet access. It writes SHA-256 checksums (verified in CI) and builds data/processed/heart_clean.csv. Unlike the common Cleveland-only approach (303 rows), ", B("all four hospitals are combined"), ", which triples the sample size. This also brings in the realistic data-quality problems described below."]),
  H2("3.2 Cleaning"),
  bullet("'?' is read as missing. The 0-4 severity score num becomes a binary target (num > 0)."),
  bullet("Physiologically impossible zeros are converted to missing: chol = 0 (all 123 Swiss rows) and trestbps = 0."),
  bullet("Out-of-range numerics and undocumented category codes are set to missing. Two exact duplicate rows are dropped."),
  bullet("No rows are dropped for missing features. Imputation happens inside the model pipeline, so serving uses identical logic."),
  ...img("reports/figures/01_class_balance.png", 6.0, "Figure 2 - Class balance overall and by hospital"),
  H2("3.3 Findings"),
  P(["The classes are balanced overall (55% positive), but the positive rate varies from 36% (Hungary) to 93% (Switzerland), so the train/test split is ", B("stratified on hospital × target"), ". Missingness is strongly site-dependent: ca is 99% missing outside Cleveland, and thal and slope are mostly missing in Hungary and VA (Figure 3). Any 'was missing' signal therefore partly encodes the hospital rather than the patient's physiology."]),
  ...img("reports/figures/05_missingness_by_site.png", 6.2, "Figure 3 - Missing values (%) per hospital"),
  ...img("reports/figures/04_categorical_disease_rate.png", 5.8, "Figure 4 - Disease rate per category (grey = missing level)"),
  P(["The strongest univariate signals are asymptomatic chest pain (cp = 4, 79% positive), exercise-induced angina, ST depression (oldpeak), male sex, a lower maximum heart rate and the number of blocked vessels (ca). Maximum heart rate falls with age (Figure 6), which motivated the engineered feature ", B("hr_reserve_pct = thalach / (220 − age)"), ": the share of age-predicted maximum heart rate the patient reached. Cholesterol and resting blood pressure have long right tails (outliers up to 603 mg/dl), which motivated a robust scaler."]),
  ...img("reports/figures/03_correlation_heatmap.png", 3.6, "Figure 5 - Spearman correlation heatmap"),
  ...img("reports/figures/07_engineered_hr_reserve.png", 6.0, "Figure 6 - Max heart rate vs age, and the engineered feature"),

  // ------------------------------------------------------------ 4 features & models
  H1("4. Feature engineering and model development"),
  H2("4.1 Feature pipeline"),
  table(["Step", "Columns", "Choice and reason"], [
    ["ClinicalFeatures", "age, thalach, oldpeak", "Adds hr_reserve_pct and age_x_oldpeak (domain features)"],
    ["Numeric", "age, trestbps, chol, thalach, oldpeak + 2 engineered", "Median imputation, then RobustScaler (outlier-tolerant)"],
    ["Categorical", "sex, cp, fbs, restecg, exang, slope, ca, thal", "Missing becomes its own level (-1), then one-hot encoding; unknown levels are ignored at serving time"],
    ["Missing indicators", "numeric", "Off by default (site proxy); tested as an ablation"],
  ], [1900, 3300, 4438]),
  gap(),
  H2("4.2 Models, tuning and selection"),
  P(["Three model families were tuned with GridSearchCV (stratified 5-fold, shuffle, seed 42) on the 80% training split, 72 configurations in total (params.yaml). ", B("Selection uses mean cross-validated ROC-AUC only"), ". The 20% hold-out set is used once, to report final performance."]),
  table(["Model", "Best params", "CV ROC-AUC", "CV acc.", "CV prec.", "CV recall", "Train-CV gap", "Test ROC-AUC"],
    order.map(k => {
      const r = models[k];
      return [nice[k], Object.entries(r.best_params).map(([a, b]) => `${a}=${b}`).join(", "),
        `${f(r.cv_roc_auc_mean)} ± ${f(r.cv_roc_auc_std)}`, f(r.cv_accuracy_mean), f(r.cv_precision_mean), f(r.cv_recall_mean),
        f(r.overfit_gap_roc_auc), f(r.test_roc_auc)];
    }), [1450, 2190, 1250, 830, 830, 900, 1000, 1188]),
  gap(),
  ...img("reports/figures/model_model_comparison.png", 6.0, "Figure 7 - Cross-validated comparison (mean ± sd over 5 folds)"),
  P([B("Why logistic regression won. "), "It has the highest CV ROC-AUC, the lowest fold-to-fold variance and a train/validation gap about 6x smaller than the tree ensembles, which overfit the 734-row training set. It also produces well-calibrated probabilities (Brier 0.123) and coefficients a clinician can read. Its L1 penalty selected C = 0.3, which shrinks weak one-hot levels to zero."]),
  H2("4.3 Decision threshold"),
  P(["For a screening tool, a missed patient costs more than a false alarm. The threshold therefore maximises ", B("F2"), " (recall weighted 2x) on out-of-fold predictions, subject to a false-alarm budget of specificity ≥ 0.70. The result is ", B(String(META.decision_threshold)), ". On the test set this gives recall ", f(T.test_recall), " (9 of 102 sick patients missed) at specificity ", f(T.test_specificity), ". The threshold is stored in the model metadata and can be overridden through a ConfigMap without rebuilding the image."]),
  ...img("reports/figures/model_champion_roc_pr.png", 5.6, "Figure 8 - Champion ROC and precision-recall curves on the hold-out test set"),
  H2("4.4 Robustness checks"),
  table(["Hospital (test)", "n", "Accuracy", "Recall", "ROC-AUC"],
    Object.entries(M.per_site_test).map(([s, v]) => [s, v.n, f(v.accuracy), f(v.recall), f(v.roc_auc)]), [3000, 1200, 1800, 1800, 1838]),
  gap(),
  P(["Performance is strongest where all features are recorded (Cleveland, ROC-AUC 0.96) and weakest where ca, thal and chol are mostly absent (VA and Switzerland, both small and imbalanced). This is a real limitation of the data and is documented rather than hidden. The ", B("missing-indicator ablation"), " gave out-of-fold ROC-AUC ", f(M.ablation_missing_indicators.oof_roc_auc_with_indicators, 4), " with indicators vs ", f(M.ablation_missing_indicators.oof_roc_auc_without_indicators, 4), " without. The difference is negligible, so the simpler model, which depends less on site artefacts, was kept. Permutation importance ranks cp, ca, fbs, slope, oldpeak and sex highest."]),

  // ------------------------------------------------------------ 5 MLflow
  H1("5. Experiment tracking (MLflow)"),
  P("Every training run creates a nested hierarchy in the heart-disease-risk experiment (SQLite backend, local artifact store):"),
  bullet([B("training-pipeline-<timestamp>"), " (parent): seed, CV setup, data SHA-256, row counts, git commit, library versions; artifacts: cleaned dataset, params.yaml, all EDA figures."]),
  bullet([B("model:<family>"), " (3 runs): best hyper-parameters, CV mean and sd for 5 metrics, train-CV gap, test metrics, ROC/PR, confusion and calibration plots, full cv_results CSV, and the logged sklearn model with signature and input example."], 1),
  bullet([B("grid:<family>:NNN"), " (72 runs): one run per hyper-parameter candidate with its CV metrics, which gives the parallel-coordinates view in Figure 10."], 1),
  bullet([B("ablation:missing_indicators"), " and ", B("champion"), ": tuned threshold, per-hospital metrics, permutation importance. The champion is registered as heart-disease-classifier v1 with alias champion and a threshold tag."], 1),
  ...img("screenshots/03_mlflow_experiment_runs.png", 5.4, "Figure 9 - MLflow run hierarchy (78 runs) and registered model"),
  ...img("screenshots/07_mlflow_rf_tuning_parallel.png", 5.6, "Figure 10 - Random forest grid: hyper-parameters vs CV score (24 child runs)"),

  // ------------------------------------------------------------ 6 packaging
  H1("6. Model packaging and reproducibility"),
  bullet([B("Formats: "), "models/model.joblib (the complete preprocessing and model Pipeline) plus model_metadata.json (version = package version + data hash, threshold, best params, CV and test metrics, git commit, training time). The model is also stored in MLflow format with code_paths, so mlflow.pyfunc.load_model('models:/heart-disease-classifier@champion') works in a new environment. Notebook 03 asserts that both artefacts give identical probabilities."]),
  bullet([B("Environment: "), "requirements.txt is fully pinned (training, tracking, tests). requirements-serve.txt is the minimal runtime used by Docker. A pyproject.toml makes the code an installable package."]),
  bullet([B("Determinism: "), "fixed seeds, checksummed raw data, the data hash logged and embedded in the model version, and every setting in params.yaml. Re-running the full pipeline in a fresh virtual environment (Prayogshala notebook dry run) reproduced identical test metrics and the same model version 1.0.0+327a95bf."]),

  // ------------------------------------------------------------ 7 CI/CD
  H1("7. CI/CD pipeline and automated testing"),
  H2("7.1 Tests"),
  table(["Test module", "What it guards"], [
    ["test_data.py (12)", "Loading all 4 sites, target mapping, sentinel zeros, invalid codes, duplicates, schema and value ranges, input not mutated"],
    ["test_model.py (13)", "Engineered features, NaN-free preprocessing, unseen categories, model learns signal (AUC > 0.8), probabilities valid, reproducibility, threshold policy, params.yaml"],
    ["test_api.py (17)", "/predict contract, confidence, high vs low risk ordering, optional fields, 422 validation, batch, /metrics, dashboard, 503 when the model is not loaded, threshold override"],
    ["test_configs.py (6)", "Kubernetes YAML, probe/port contract with the Dockerfile, monitoring copies in sync, dashboard queries, alert rules, CI job order"],
    ["test_pipeline_smoke.py (2)", "EDA figures and a miniature end-to-end training run with an isolated MLflow store (all run types present)"],
  ], [2500, 7138]),
  gap(),
  ...img("screenshots/01_tests_terminal.png", 4.0, "Figure 11 - flake8 and pytest in a clean virtual environment: 50 passed, 97% coverage"),
  H2("7.2 GitHub Actions workflow"),
  P(".github/workflows/ci-cd.yml runs on every push and pull request. Each job depends on the previous one, so any failure stops the pipeline. Shell steps use pipefail so that errors are not hidden by tee."),
  table(["Job", "Steps", "Artifacts / logs"], [
    ["lint", "flake8 on src, tests, scripts, kubeflow", "flake8-report.txt"],
    ["test", "clean pip install from requirements.txt, dataset checksum verification, pytest with JUnit and coverage", "junit.xml, coverage.xml, HTML coverage, pytest.log, summary table"],
    ["train", "data script, EDA, MLflow training, quality gate (test ROC-AUC ≥ 0.85 or the job fails)", "model, reports and figures, mlflow.db + mlartifacts, metrics table"],
    ["docker", "build with git sha label, run, wait for /ready, smoke-test /predict (asserts classes), batch, 422 check, /metrics, health status, push to GHCR on main", "build log, container log, prediction JSON, image tar"],
    ["deploy-kind", "kind cluster, load image, kubectl apply -k, rollout status, port-forward, /ready and /predict, 100-request load test", "kubectl resources, describe, pod logs"],
  ], [1400, 5200, 3038]),
  gap(),
  note("CI evidence: after pushing the repository, add screenshots of the Actions run graph and job summaries here (screenshots/02_ci_pipeline.png, 02b_ci_summary.png). The workflow was validated locally (YAML structure test, every command run by hand), but it has not yet run on GitHub."),

  // ------------------------------------------------------------ 8 container & deploy
  H1("8. Containerisation and deployment"),
  H2("8.1 Docker image"),
  P("A multi-stage Dockerfile builds a virtual environment that contains only the serving dependencies, then copies it into a slim runtime stage. The container runs as non-root uid 10001, defines a HEALTHCHECK on /health, carries OCI labels (git sha, build date), and starts uvicorn. The API accepts JSON and returns the class, probability, confidence (the probability of the predicted class), a risk band, and the optional inputs that were imputed. Pydantic rejects out-of-range or unknown fields with HTTP 422."),
  ...img("screenshots/09_docker_terminal.png", 6.2, "Figure 12 - Image build, healthy container and sample predictions (high, low, partial, invalid)"),
  H2("8.2 Kubernetes"),
  P(["Manifests are organised with kustomize (k8s/base, a k3s overlay that switches the ingress class to Traefik, and k8s/monitoring). The Deployment runs 2 replicas with a rolling update (maxUnavailable 0), startup, readiness and liveness probes, CPU and memory requests and limits, a read-only root filesystem with all capabilities dropped, Prometheus scrape annotations, and a preStop drain hook. The HPA scales between 2 and 6 pods at 70% CPU, and a PodDisruptionBudget keeps at least one pod available. The API is exposed through a LoadBalancer Service and an Ingress (heart-risk.local)."]),
  P(["The deployment was verified on a local single-node Kubernetes cluster (k3s v1.31). The HPA scaled the API to 4 replicas under load. Verification found two real issues, both fixed and rolled out as new image versions: (1) an empty DECISION_THRESHOLD value in the ConfigMap crashed predictions, so it is now treated as unset and covered by a test; (2) about 1% of requests returned 502 during a rollout until the preStop drain hook was added. ", B("After the fix, 1,111 of 1,111 requests sent through the Ingress during a rolling restart succeeded.")]),
  ...img("screenshots/17_k8s_terminal.png", 6.3, "Figure 13 - Deployment, pods, Service, Ingress, HPA, PDB and rollout history"),
  ...img("screenshots/18_k8s_verify_terminal.png", 6.0, "Figure 14 - Endpoint checks via Ingress and Service, load test and zero-downtime rollout"),
  P(["Access instructions: kubectl apply -k k8s/base (or k8s/overlays/k3s), then use the LoadBalancer IP (cloud, Docker Desktop, or minikube tunnel), the Ingress host heart-risk.local, or kubectl -n heart-risk port-forward svc/heart-risk-api 8080:80. For a public cloud (GKE, EKS or AKS) the same manifests apply with the GHCR image that CI pushes."]),

  // ------------------------------------------------------------ 9 monitoring
  H1("9. Monitoring and logging"),
  bullet([B("Request logging: "), "every request produces one JSON log line with request_id (also returned in the x-request-id header), method, path, status, latency and client. Predictions and validation errors are logged as separate events."]),
  bullet([B("Metrics (/metrics): "), "request counter by endpoint and status, latency histogram, in-flight gauge, predictions by outcome and risk band, predicted-probability histogram, imputed inputs per feature, validation errors, model info."]),
  bullet([B("Prometheus "), "discovers every API pod through the Kubernetes API (RBAC-scoped Role) and scrapes it every 5 seconds. Five alert rules cover: target down, 5xx ratio > 5%, p95 latency > 250 ms, prediction drift (positive rate moves more than 25 points from the training base rate of 0.55), and imputation-heavy traffic. promtool validated the rules."]),
  bullet([B("Grafana "), "dashboard (provisioned from JSON, 12 panels): targets up, traffic, p50/p95/p99 latency, risk-band mix, drift line against the training base rate, probability distribution and missing-input counts. All 15 dashboard queries were executed against the live Prometheus and returned data. The built-in /monitoring page gives a lightweight view without Grafana."]),
  ...img("screenshots/13_prometheus_targets.png", 4.6, "Figure 15 - Prometheus discovering and scraping all four API pods"),
  ...img("screenshots/14_prometheus_rate_latency.png", 3.4, "Figure 16 - Per-pod request rate and p95 /predict latency under load"),
  ...img("screenshots/12_api_live_monitor.png", 4.6, "Figure 17 - Built-in live monitor (/monitoring)"),

  // ------------------------------------------------------------ 10 prayogshala
  H1("10. Running in BITS Prayogshala"),
  P(["notebooks/00_prayogshala_end_to_end.ipynb reproduces the whole pipeline inside a Kubeflow notebook server: it locates the uploaded project (zip or git clone), installs the pinned requirements, verifies checksums, runs flake8 and pytest, runs EDA and MLflow training, queries the registry, starts the MLflow UI and the API as background processes, calls /predict, generates traffic and reads /metrics, checks Kubernetes permissions with kubectl auth can-i, and cleans up. A dry run in a brand-new virtual environment completed in about 2.5 minutes on 2 CPUs. kubeflow/pipeline.py (KFP v2, compiled and validated) runs the same steps as a pipeline: fetch-and-test, train-and-track (exports metrics and the model artefact), quality-gate, and a conditional publish-model step."]),

  // ------------------------------------------------------------ 11 conclusion
  H1("11. Limitations and next steps"),
  bullet("The data is small (918 patients) and decades old. Performance is weaker for hospitals with heavily missing ca/thal. This is a screening aid, not a diagnostic tool."),
  bullet("Drift detection currently watches the output distribution. A next step is feature-level drift (e.g. PSI per input) computed from logged requests, plus scheduled re-training through the Kubeflow pipeline."),
  bullet("MLflow uses a local SQLite store. A shared tracking server with object storage would suit a team setting."),
  H1("12. Deliverables and links"),
  table(["Deliverable", "Location"], [
    ["Code repository", REPO_URL],
    ["Dataset + download script", "data/raw, data/processed/heart_clean.csv, scripts/download_data.py"],
    ["Notebooks", "notebooks/00_prayogshala_end_to_end, 01_eda, 02_training, 03_inference"],
    ["Tests / CI", "tests/, .github/workflows/ci-cd.yml"],
    ["Docker / Kubernetes", "Dockerfile, docker-compose.yml, k8s/base, k8s/overlays/k3s, k8s/monitoring"],
    ["Monitoring", "monitoring/ (Prometheus config, alert rules, Grafana dashboard)"],
    ["Screenshots", "screenshots/"],
    ["Video", "report/VIDEO_SCRIPT.md (recording walkthrough)"],
  ], [2800, 6838]),
];

const doc = new Document({
  creator: "Shankaraguru Ramdurg",
  title: "MLOps Assignment 1 - Heart Disease Risk Prediction",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1F3A5F", font: FONT }, paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "2E5A88", font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 900, hanging: 260 } } } },
  ] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "AIMLCZG523 Assignment 1 - Heart Disease Risk MLOps - 2025AE05523", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: "888888" })] })] }) },
    children,
  }],
});

const out = path.join(__dirname, "MLOps_Assignment1_Report_2025AE05523.docx");
Packer.toBuffer(doc).then(b => { fs.writeFileSync(out, b); console.log("wrote", out); });
