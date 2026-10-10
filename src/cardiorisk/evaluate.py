"""Metrics and evaluation plots shared by training and notebooks."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, proba) if len(set(y_true)) > 1 else float("nan"),
        "brier": brier_score_loss(y_true, proba),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def choose_threshold(y_true, proba, beta: float = 2.0, lo: float = 0.25, hi: float = 0.65,
                     min_specificity: float = 0.0) -> float:
    """Threshold in [lo, hi] maximising F-beta on (out-of-fold) predictions.

    Thresholds whose specificity falls below `min_specificity` are excluded
    (a "false-alarm budget"); if none qualify the highest threshold is used.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    grid = np.round(np.arange(lo, hi + 1e-9, 0.01), 2)
    best_t, best_s = float(hi), -1.0
    for t in grid:
        pred = (proba >= t).astype(int)
        neg = y_true == 0
        spec = float((pred[neg] == 0).mean()) if neg.any() else 1.0
        if spec < min_specificity:
            continue
        s = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t


# ----------------------------------------------------------------- plots
def plot_roc_pr(y_true, proba, title: str, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend(loc="lower right")
    prec, rec, _ = precision_recall_curve(y_true, proba)
    axes[1].plot(rec, prec, lw=2, label=f"AP = {average_precision_score(y_true, proba):.3f}")
    axes[1].axhline(np.mean(y_true), ls="--", c="k", lw=1, label="baseline")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend(loc="lower left")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_confusion(y_true, proba, threshold: float, title: str, path: Path) -> Path:
    pred = (np.asarray(proba) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, pred), display_labels=["No disease", "Disease"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{title}\n(threshold = {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_calibration(y_true, proba, title: str, path: Path) -> Path:
    frac, mean_pred = calibration_curve(y_true, proba, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(mean_pred, frac, "o-", label="model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", title=f"Calibration - {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def feature_importance(pipeline, X: pd.DataFrame, y, n_repeats: int = 10, seed: int = 42) -> pd.Series:
    """Model-agnostic permutation importance on raw input columns (ROC-AUC drop)."""
    from sklearn.inspection import permutation_importance

    r = permutation_importance(pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=seed, n_jobs=1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values()


def plot_importance(imp: pd.Series, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.6))
    colors = ["#d8573b" if v > 0 else "#999999" for v in imp.values]
    ax.barh(imp.index, imp.values, color=colors)
    ax.set(xlabel="Mean ROC-AUC drop when permuted", title=f"Permutation importance - {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> Path:
    """summary: index=model, columns cv_<metric>_mean / cv_<metric>_std."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    w = 0.8 / len(summary)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (name, row) in enumerate(summary.iterrows()):
        means = [row[f"cv_{m}_mean"] for m in metrics]
        stds = [row[f"cv_{m}_std"] for m in metrics]
        ax.bar(x + i * w, means, w, yerr=stds, capsize=3, label=name)
    ax.set_xticks(x + w * (len(summary) - 1) / 2, metrics)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("5-fold CV score (mean +/- sd)")
    ax.set_title("Model comparison on cross-validation", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
