"""
Data drift detection: PSI, KS test, KL divergence, chi-square for categoricals.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class DriftResult:
    feature: str
    method: str
    statistic: float
    p_value: float | None
    drifted: bool
    severity: str  # "none", "low", "medium", "high"


def _severity(score: float, thresholds: tuple = (0.1, 0.2, 0.25)) -> str:
    if score < thresholds[0]:
        return "none"
    if score < thresholds[1]:
        return "low"
    if score < thresholds[2]:
        return "medium"
    return "high"


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index."""
    ref_min, ref_max = reference.min(), reference.max()
    if ref_max == ref_min:
        return 0.0
    edges = np.linspace(ref_min, ref_max, bins + 1)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = (ref_counts + 1e-6) / len(reference)
    cur_pct = (cur_counts + 1e-6) / len(current)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    stat, p = stats.ks_2samp(reference, current)
    return float(stat), float(p)


def kl_divergence(reference: np.ndarray, current: np.ndarray, bins: int = 20) -> float:
    edges = np.histogram_bin_edges(np.concatenate([reference, current]), bins=bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = (ref_counts + 1e-9) / ref_counts.sum()
    cur_pct = (cur_counts + 1e-9) / cur_counts.sum()
    return float(stats.entropy(cur_pct, ref_pct))


def chi_square_test(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    cats = list(set(reference.unique()) | set(current.unique()))
    ref_counts = reference.value_counts().reindex(cats, fill_value=0)
    cur_counts = current.value_counts().reindex(cats, fill_value=0)
    stat, p = stats.chisquare(cur_counts + 1, ref_counts + 1)
    return float(stat), float(p)


def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.05,
) -> list[DriftResult]:
    results = []
    for col in reference.columns:
        if col not in current.columns:
            continue
        ref_col = reference[col].dropna()
        cur_col = current[col].dropna()

        if ref_col.dtype == object or ref_col.nunique() < 10:
            stat, p = chi_square_test(ref_col, cur_col)
            drifted = p < ks_alpha
            results.append(DriftResult(
                feature=col, method="chi-square",
                statistic=stat, p_value=p,
                drifted=drifted,
                severity="high" if drifted else "none"
            ))
        else:
            ref_arr = ref_col.values.astype(float)
            cur_arr = cur_col.values.astype(float)
            psi_val = psi(ref_arr, cur_arr)
            ks_stat, ks_p = ks_test(ref_arr, cur_arr)
            drifted = psi_val > psi_threshold or ks_p < ks_alpha
            results.append(DriftResult(
                feature=col, method="PSI+KS",
                statistic=psi_val, p_value=ks_p,
                drifted=drifted,
                severity=_severity(psi_val)
            ))
    return results


def drift_summary(results: list[DriftResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Feature": r.feature, "Method": r.method,
        "Statistic": round(r.statistic, 4),
        "P-Value": round(r.p_value, 4) if r.p_value is not None else "-",
        "Drifted": "YES" if r.drifted else "no",
        "Severity": r.severity,
    } for r in results])
