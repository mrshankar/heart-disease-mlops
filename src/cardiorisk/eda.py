"""Exploratory data analysis - produces the figures used in the report.

Run:  python -m cardiorisk.eda
Figures are written to reports/figures/ and a JSON summary to
reports/eda_summary.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config  # noqa: E402
from .data import load_clean, missing_report  # noqa: E402

log = logging.getLogger(__name__)

PALETTE = {0: "#3b7dd8", 1: "#d8573b"}
LABELS = {0: "No disease", 1: "Disease"}
NICE = {
    "age": "Age (years)", "trestbps": "Resting BP (mm Hg)", "chol": "Cholesterol (mg/dl)",
    "thalach": "Max heart rate", "oldpeak": "ST depression (oldpeak)",
    "sex": "Sex (1 = male)", "cp": "Chest pain type", "fbs": "Fasting sugar > 120",
    "restecg": "Resting ECG", "exang": "Exercise angina", "slope": "ST slope",
    "ca": "Major vessels (ca)", "thal": "Thalassemia",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 2]})
    counts = df[config.TARGET].value_counts().sort_index()
    axes[0].bar([LABELS[i] for i in counts.index], counts.values, color=[PALETTE[i] for i in counts.index])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, f"{v} ({v / len(df):.0%})", ha="center")
    axes[0].set_ylim(0, counts.max() * 1.15)
    axes[0].set_title("Overall class balance")
    axes[0].set_ylabel("Patients")

    ct = pd.crosstab(df[config.SOURCE_COL], df[config.TARGET])
    ct.columns = [LABELS[c] for c in ct.columns]
    ct.plot(kind="barh", stacked=True, ax=axes[1], color=[PALETTE[0], PALETTE[1]])
    for i, (src, row) in enumerate(ct.iterrows()):
        axes[1].text(row.sum() + 3, i, f"{row['Disease'] / row.sum():.0%} positive", va="center", fontsize=9)
    axes[1].set_title("Class balance by hospital")
    axes[1].set_xlabel("Patients")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, ct.sum(axis=1).max() * 1.3)
    fig.subplots_adjust(wspace=0.55)
    return _save(fig, out_dir, "01_class_balance.png")


def plot_numeric_histograms(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.6))
    for ax, col in zip(axes, cols):
        for t in (0, 1):
            sns.histplot(df.loc[df[config.TARGET] == t, col].dropna(), ax=ax, color=PALETTE[t],
                         label=LABELS[t], kde=True, stat="density", alpha=0.35, bins=25)
        ax.set_title(NICE[col])
        ax.set_xlabel("")
    axes[0].legend()
    fig.suptitle("Numeric feature distributions by outcome", y=1.04, fontweight="bold")
    return _save(fig, out_dir, "02_numeric_histograms.png")


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> Path:
    corr = df[config.INPUT_FEATURES + [config.TARGET]].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.7}, ax=ax, annot_kws={"size": 8})
    ax.set_title("Spearman correlation (pairwise complete observations)")
    return _save(fig, out_dir, "03_correlation_heatmap.png")


def plot_categorical_rates(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = config.CATEGORICAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), cols):
        s = df[col].map(lambda v: "missing" if pd.isna(v) else str(int(v)))
        g = df.groupby(s)[config.TARGET].agg(["mean", "size"])
        bars = ax.bar(g.index, g["mean"], color=["#999999" if i == "missing" else "#d8573b" for i in g.index])
        for b, n in zip(bars, g["size"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.axhline(df[config.TARGET].mean(), ls="--", c="k", lw=1)
        ax.set_ylim(0, 1.1)
        ax.set_title(NICE[col])
        ax.set_ylabel("Disease rate" if ax in axes[:, 0] else "")
    fig.suptitle("Disease rate per category (dashed = overall rate; grey = missing)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "04_categorical_disease_rate.png")


def plot_missingness(df: pd.DataFrame, out_dir: Path) -> Path:
    rep = missing_report(df)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    sns.heatmap(rep, annot=True, fmt=".0f", cmap="Greys", cbar_kws={"label": "% missing"}, ax=ax)
    ax.set_title("Missing values (%) per hospital - missingness is site-dependent")
    ax.set_ylabel("")
    return _save(fig, out_dir, "05_missingness_by_site.png")


def plot_boxplots(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = config.NUMERIC_FEATURES
    long = df.melt(id_vars=[config.TARGET], value_vars=cols)
    long[config.TARGET] = long[config.TARGET].map(LABELS)
    g = sns.catplot(data=long, x=config.TARGET, y="value", col="variable", kind="box", sharey=False,
                    height=3.4, aspect=0.8, palette=[PALETTE[0], PALETTE[1]], hue=config.TARGET, legend=False)
    g.set_titles("{col_name}")
    g.set_xlabels("")
    g.figure.suptitle("Outliers and spread by outcome", y=1.05, fontweight="bold")
    path = out_dir / "06_boxplots.png"
    g.figure.savefig(path)
    plt.close(g.figure)
    return path


def plot_age_hr(df: pd.DataFrame, out_dir: Path) -> Path:
    """Motivates the engineered feature hr_reserve_pct = thalach / (220 - age)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for t in (0, 1):
        d = df[df[config.TARGET] == t]
        axes[0].scatter(d["age"], d["thalach"], s=12, alpha=0.5, c=PALETTE[t], label=LABELS[t])
    xs = np.linspace(25, 80, 10)
    axes[0].plot(xs, 220 - xs, "k--", lw=1, label="220 - age (predicted max)")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Max heart rate achieved")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Max heart rate vs age")
    pct = df["thalach"] / (220 - df["age"])
    for t in (0, 1):
        sns.kdeplot(pct[df[config.TARGET] == t].dropna(), ax=axes[1], fill=True, color=PALETTE[t], label=LABELS[t])
    axes[1].set_xlabel("thalach / (220 - age)")
    axes[1].set_title("Engineered: % of age-predicted max HR")
    axes[1].legend()
    return _save(fig, out_dir, "07_engineered_hr_reserve.png")


def summarize(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "positive_rate": round(float(df[config.TARGET].mean()), 4),
        "rows_per_source": df[config.SOURCE_COL].value_counts().to_dict(),
        "positive_rate_per_source": df.groupby(config.SOURCE_COL)[config.TARGET].mean().round(3).to_dict(),
        "missing_pct": (df[config.INPUT_FEATURES].isna().mean() * 100).round(1).to_dict(),
        "numeric_describe": df[config.NUMERIC_FEATURES].describe().round(2).to_dict(),
        "top_abs_spearman_with_target": df[config.INPUT_FEATURES + [config.TARGET]]
        .corr(method="spearman")[config.TARGET].drop(config.TARGET).abs().sort_values(ascending=False)
        .round(3).head(6).to_dict(),
    }


def run_eda(out_dir: Path | str = config.FIGURES_DIR) -> dict:
    _style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    paths = [f(df, out_dir) for f in (plot_class_balance, plot_numeric_histograms, plot_correlation,
                                      plot_categorical_rates, plot_missingness, plot_boxplots, plot_age_hr)]
    summary = summarize(df)
    summary["figures"] = [p.name for p in paths]
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("EDA: wrote %d figures to %s", len(paths), out_dir)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    print(json.dumps(run_eda(), indent=2, default=str)[:1500])
