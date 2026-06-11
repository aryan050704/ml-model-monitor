"""
Track model performance metrics over time windows.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)


@dataclass
class WindowMetrics:
    window_id: int
    start_idx: int
    end_idx: int
    metrics: dict[str, float]


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return {k: round(v, 4) for k, v in m.items()}


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
        "mape": round(float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9)))) * 100, 4),
    }


def sliding_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "classification",
    window_size: int = 100,
    step: int = 50,
    y_prob: np.ndarray | None = None,
) -> list[WindowMetrics]:
    results = []
    i = 0
    win_id = 0
    while i + window_size <= len(y_true):
        yt = y_true[i: i + window_size]
        yp = y_pred[i: i + window_size]
        yprob = y_prob[i: i + window_size] if y_prob is not None else None
        if task == "classification":
            m = compute_classification_metrics(yt, yp, yprob)
        else:
            m = compute_regression_metrics(yt, yp)
        results.append(WindowMetrics(window_id=win_id, start_idx=i, end_idx=i + window_size, metrics=m))
        i += step
        win_id += 1
    return results


def metrics_to_df(windows: list[WindowMetrics]) -> pd.DataFrame:
    rows = []
    for w in windows:
        row = {"window": w.window_id, "start": w.start_idx, "end": w.end_idx}
        row.update(w.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def detect_performance_degradation(df: pd.DataFrame, metric: str, threshold: float = 0.05) -> pd.DataFrame:
    baseline = df[metric].iloc[0]
    df = df.copy()
    df["delta"] = df[metric] - baseline
    df["degraded"] = df["delta"] < -threshold
    return df
