"""Calibration metrics and reliability plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_BINS = np.linspace(0.0, 1.0, 11)  # deciles


def brier_score(prob: pd.Series | np.ndarray, outcome: pd.Series | np.ndarray) -> float:
    """Mean squared error between probability and binary outcome."""
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    mask = ~np.isnan(p) & ~np.isnan(y)
    if not mask.any():
        return float("nan")
    return float(np.mean((p[mask] - y[mask]) ** 2))


def calibration_table(df: pd.DataFrame, bins: np.ndarray = DEFAULT_BINS) -> pd.DataFrame:
    """Per-bucket calibration: bin Polymarket probabilities, then summarise.

    Columns
    -------
    bin_left, bin_right : edges of the bucket (poly_prob)
    n                   : count
    poly_mean, noaa_mean, outcome_rate
    abs_gap             : |poly_mean - noaa_mean|
    flagged             : abs_gap > 0.15
    """
    df = df.dropna(subset=["poly_prob", "noaa_prob", "outcome"]).copy()
    df["bucket"] = pd.cut(df["poly_prob"], bins=bins, include_lowest=True)
    grouped = df.groupby("bucket", observed=True).agg(
        n=("outcome", "size"),
        poly_mean=("poly_prob", "mean"),
        noaa_mean=("noaa_prob", "mean"),
        outcome_rate=("outcome", "mean"),
    )
    grouped = grouped.reset_index()
    grouped["bin_left"] = [float(b.left) for b in grouped["bucket"]]
    grouped["bin_right"] = [float(b.right) for b in grouped["bucket"]]
    grouped["abs_gap"] = (grouped["poly_mean"] - grouped["noaa_mean"]).abs()
    grouped["flagged"] = grouped["abs_gap"] > 0.15
    return grouped.drop(columns="bucket")


def plot_reliability(df: pd.DataFrame, out_path: Path | str) -> Path:
    """Reliability diagram: predicted probability vs realised frequency."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")

    for label, col, marker in (
        ("Polymarket J-1", "poly_prob", "o"),
        ("NOAA", "noaa_prob", "s"),
    ):
        sub = df.dropna(subset=[col, "outcome"])
        if sub.empty:
            continue
        bins = DEFAULT_BINS
        sub = sub.assign(_bucket=pd.cut(sub[col], bins=bins, include_lowest=True))
        agg = sub.groupby("_bucket", observed=True).agg(
            x=(col, "mean"), y=("outcome", "mean"), n=("outcome", "size")
        )
        ax.plot(agg["x"], agg["y"], marker=marker, label=label)
        for x, y, n in zip(agg["x"], agg["y"], agg["n"]):
            ax.annotate(f"n={n}", (x, y), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Realised frequency")
    ax.set_title("Reliability diagram — Polymarket vs NOAA")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
