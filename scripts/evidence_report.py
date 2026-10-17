#!/usr/bin/env python
"""Build reports/evidence_report.html - one self-contained page with every result
of a pipeline run (for screenshots, the written report and the demo video).

    python scripts/evidence_report.py                 # uses the latest logs/<timestamp>/
    python scripts/evidence_report.py --logs logs/20261017-140012

Everything is embedded (images as base64), so the file opens in any browser,
can be copied off the VM, and needs no server. Standard library only.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import platform
import re
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "evidence_report.html"

EDA_CAPTIONS = {
    "01_class_balance.png": "Class balance overall and by hospital",
    "02_numeric_histograms.png": "Numeric feature distributions by outcome",
    "03_correlation_heatmap.png": "Spearman correlation heatmap",
    "04_categorical_disease_rate.png": "Disease rate per category (grey = missing)",
    "05_missingness_by_site.png": "Missing values (%) per hospital",
    "06_boxplots.png": "Outliers and spread by outcome",
    "07_engineered_hr_reserve.png": "Engineered feature: % of age-predicted max heart rate",
}
MODEL_CAPTIONS = {
    "model_model_comparison.png": "5-fold CV comparison of the three model families",
    "model_champion_roc_pr.png": "Champion ROC and precision-recall (hold-out test)",
    "model_champion_confusion.png": "Champion confusion matrix at the tuned threshold",
    "model_champion_calibration.png": "Champion calibration",
    "model_champion_permutation_importance.png": "Permutation importance",
}


def esc(x) -> str:
    return html.escape(str(x))


def img(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{esc(caption)}">'
            f"<figcaption>{esc(caption)} <span class=muted>({esc(path.name)})</span></figcaption></figure>")


def read(p: Path) -> str:
    return p.read_text(errors="replace") if p.exists() else ""


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def table(headers, rows) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def pre(text: str, max_lines: int = 60) -> str:
    lines = strip_ansi(text).rstrip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines in the log file)"]
    return f"<pre>{esc(chr(10).join(lines))}</pre>"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "log", "--oneline", "-1"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def mlflow_summary(db: Path) -> str:
    if not db.exists():
        return "<p class=muted>mlflow.db not found - run the train stage.</p>"
    con = sqlite3.connect(db)
    try:
        kinds = con.execute(
            "SELECT t.value, COUNT(*) FROM tags t JOIN runs r ON r.run_uuid = t.run_uuid "
            "WHERE t.key = 'run_type' AND r.lifecycle_stage = 'active' GROUP BY t.value ORDER BY 2 DESC").fetchall()
        total = sum(n for _, n in kinds)
        versions = con.execute("SELECT name, version, run_id FROM model_versions ORDER BY version").fetchall()
        aliases = con.execute("SELECT name, alias, version FROM registered_model_aliases").fetchall()
    finally:
        con.close()
    out = f"<p><b>{total}</b> tracked runs in mlflow.db</p>"
    out += table(["run type", "runs"], kinds)
    out += "<h3>Model registry</h3>" + table(["model", "version", "run id"], versions)
    if aliases:
        out += table(["model", "alias", "version"], aliases)
    return out


def tests_summary(log: str) -> str:
    log = strip_ansi(log)
    passed = re.search(r"(\d+) passed.*?in [\d.]+s", log)
    failed = re.search(r"(\d+) failed", log)
    cov = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+%)", log, re.M)
    lint = "flake8: no issues" in log
    cells = [("flake8", "clean" if lint else "see log"),
             ("tests passed", passed.group(1) if passed else "?"),
             ("tests failed", failed.group(1) if failed else "0"),
             ("line coverage", cov.group(1) if cov else "?")]
    return "".join(f'<div class=tile><div class=k>{esc(k)}</div><div class=v>{esc(v)}</div></div>' for k, v in cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, help="logs/<timestamp> directory (default: latest)")
    args = ap.parse_args()
    logs_root = ROOT / "logs"
    log_dir = args.logs or (max((p for p in logs_root.iterdir() if p.is_dir()), default=None)
                            if logs_root.exists() else None)
    L = (lambda name: read(log_dir / f"{name}.log")) if log_dir else (lambda name: "")

    meta = json.loads(read(ROOT / "models/model_metadata.json") or "{}")
    metrics = json.loads(read(ROOT / "reports/metrics.json") or "{}")
    t = meta.get("test_metrics", {})

    sections = []
    # ---- run summary
    summary = strip_ansi(read(log_dir / "summary.txt")) if log_dir else ""
    sections.append(("Pipeline run", f"""
      <p>Commit <code>{esc(git_commit())}</code> &middot; host <code>{esc(platform.node())}</code> &middot;
      Python {esc(platform.python_version())} &middot; logs <code>{esc(log_dir.relative_to(ROOT) if log_dir else '-')}</code></p>
      {pre(summary) if summary else '<p class=muted>No summary.txt (run ./run_all.sh).</p>'}"""))

    # ---- key results
    if meta:
        tiles = [("champion model", meta.get("model_family")), ("decision threshold", meta.get("decision_threshold")),
                 ("CV ROC-AUC", f"{meta['cv_metrics']['cv_roc_auc_mean']:.3f} ± {meta['cv_metrics']['cv_roc_auc_std']:.3f}"),
                 ("test ROC-AUC", t.get("test_roc_auc")), ("test recall", t.get("test_recall")),
                 ("test specificity", t.get("test_specificity")), ("test accuracy", t.get("test_accuracy")),
                 ("model version", meta.get("model_version"))]
        body = "<div class=tiles>" + "".join(
            f'<div class=tile><div class=k>{esc(k)}</div><div class=v>{esc(v)}</div></div>' for k, v in tiles) + "</div>"
        if metrics.get("models"):
            rows = [[k, v.get("best_params"), v.get("cv_roc_auc_mean"), v.get("cv_roc_auc_std"), v.get("cv_accuracy_mean"),
                     v.get("cv_recall_mean"), v.get("overfit_gap_roc_auc"), v.get("test_roc_auc")]
                    for k, v in metrics["models"].items()]
            body += "<h3>Model comparison (selection on CV ROC-AUC)</h3>" + table(
                ["model", "best params", "CV ROC-AUC", "sd", "CV acc", "CV recall", "train-CV gap", "test ROC-AUC"], rows)
        if metrics.get("per_site_test"):
            body += "<h3>Hold-out test per hospital</h3>" + table(
                ["hospital", "n", "accuracy", "recall", "ROC-AUC"],
                [[s, v["n"], v["accuracy"], v["recall"], v["roc_auc"]] for s, v in metrics["per_site_test"].items()])
        sections.append(("Results", body))

    # ---- EDA + model plots
    sections.append(("EDA plots", "<div class=grid>" + "".join(img(FIG / f, c) for f, c in EDA_CAPTIONS.items()) + "</div>"))
    sections.append(("Model evaluation plots",
                     "<div class=grid>" + "".join(img(FIG / f, c) for f, c in MODEL_CAPTIONS.items()) + "</div>"))

    # ---- tests, MLflow, serving, docker, k8s
    q = L("quality")
    sections.append(("Lint and tests", f"<div class=tiles>{tests_summary(q)}</div>" + pre(
        "\n".join(ln for ln in strip_ansi(q).splitlines() if not ln.startswith("....")), 40)))
    sections.append(("Experiment tracking (MLflow)", mlflow_summary(ROOT / "mlflow.db")))
    for name, title in [("serve", "API serving: predictions, validation, load test, metrics"),
                        ("docker", "Docker image and container smoke test"), ("k8s", "Kubernetes deployment")]:
        txt = L(name)
        if txt:
            sections.append((title, pre(txt, 80)))
    tr = L("train")
    if tr:
        sections.append(("Training log (excerpt)", pre("\n".join(
            ln for ln in strip_ansi(tr).splitlines() if "cardiorisk" in ln or "champion" in ln.lower()
            or "quality gate" in ln or "test ROC-AUC" in ln), 40)))

    nav = "".join(f'<a href="#s{i}">{esc(t_)}</a>' for i, (t_, _) in enumerate(sections))
    body = "".join(f'<section id="s{i}"><h2>{esc(t_)}</h2>{c}</section>' for i, (t_, c) in enumerate(sections))
    OUT.write_text(f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Heart Disease MLOps - run evidence</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#1d2530;--muted:#6b7480;--line:#e3e6ea;--acc:#1f3a5f}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{background:var(--acc);color:#fff;padding:18px 28px}} header h1{{margin:0;font-size:20px}} header p{{margin:4px 0 0;opacity:.85}}
nav{{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:8px 28px;display:flex;gap:14px;flex-wrap:wrap;z-index:2}}
nav a{{color:var(--acc);text-decoration:none;font-size:13px}} main{{max-width:1200px;margin:auto;padding:12px 28px 60px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:16px 0}}
h2{{margin:0 0 10px;font-size:17px;color:var(--acc)}} h3{{font-size:14px;margin:16px 0 6px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:8px}}
.tile{{border:1px solid var(--line);border-radius:8px;padding:10px 12px}} .tile .k{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
.tile .v{{font-size:19px;font-weight:650;font-variant-numeric:tabular-nums;word-break:break-word}}
table{{border-collapse:collapse;width:100%;margin:6px 0;font-variant-numeric:tabular-nums}} th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left}}
th{{background:#f0f3f7}} pre{{background:#0e1116;color:#d6dde6;padding:12px 14px;border-radius:8px;overflow:auto;font-size:12px;max-height:520px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px}} figure{{margin:0}}
img{{width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}} figcaption{{font-size:12.5px;margin-top:4px}}
.muted{{color:var(--muted)}} code{{background:#eef1f5;padding:1px 5px;border-radius:4px}}
</style></head><body>
<header><h1>Heart Disease Risk - MLOps pipeline run evidence</h1>
<p>AIMLCZG523 Assignment 1 &middot; Shankaraguru Ramdurg (2025AE05523) &middot; generated {datetime.now():%Y-%m-%d %H:%M}</p></header>
<nav>{nav}</nav><main>{body}</main></body></html>""")
    print(f"evidence report: {OUT.relative_to(ROOT)}  (open it in a browser)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
