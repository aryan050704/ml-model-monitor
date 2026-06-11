import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from monitor.drift import detect_feature_drift, drift_summary
from monitor.performance import (
    compute_classification_metrics, sliding_window_metrics,
    metrics_to_df, detect_performance_degradation
)
from monitor.alerts import AlertManager
from demo.generate_data import make_classification_data

st.set_page_config(page_title="ML Model Monitor", layout="wide")
st.title("ML Model Monitor")
st.markdown("Production-grade **data drift detection**, **performance monitoring**, and **alerting** for ML models.")

# --- Session ---
if "alert_mgr" not in st.session_state:
    st.session_state.alert_mgr = AlertManager()

# --- Sidebar ---
st.sidebar.header("Data Source")
use_demo = st.sidebar.checkbox("Use demo data (synthetic drift)", value=True)
psi_threshold = st.sidebar.slider("PSI drift threshold", 0.05, 0.5, 0.2, 0.05)
ks_alpha = st.sidebar.slider("KS test α", 0.01, 0.1, 0.05, 0.01)
window_size = st.sidebar.slider("Performance window size", 50, 500, 100, 50)
perf_metric = st.sidebar.selectbox("Primary metric", ["accuracy", "f1", "precision", "recall"])

if st.sidebar.button("Reset Alerts"):
    st.session_state.alert_mgr.clear()
    st.sidebar.success("Alerts cleared.")

# --- Load data ---
if use_demo:
    ref_df, prod_df, ref_true, ref_pred, prod_true, prod_pred = make_classification_data(drift=True)
else:
    col1, col2 = st.columns(2)
    ref_file = col1.file_uploader("Reference dataset CSV", type="csv", key="ref")
    prod_file = col2.file_uploader("Production dataset CSV", type="csv", key="prod")
    ref_pred_file = col1.file_uploader("Reference predictions CSV (columns: true, pred)", type="csv", key="rp")
    prod_pred_file = col2.file_uploader("Production predictions CSV (columns: true, pred)", type="csv", key="pp")

    if not all([ref_file, prod_file, ref_pred_file, prod_pred_file]):
        st.info("Upload all 4 CSV files or enable demo mode.")
        st.stop()

    ref_df = pd.read_csv(ref_file)
    prod_df = pd.read_csv(prod_file)
    rp = pd.read_csv(ref_pred_file)
    pp = pd.read_csv(prod_pred_file)
    ref_true, ref_pred = rp["true"].values, rp["pred"].values
    prod_true, prod_pred = pp["true"].values, pp["pred"].values

# ---- DRIFT ANALYSIS ----
st.header("Data Drift Analysis")
numeric_ref = ref_df.select_dtypes(include=np.number)
numeric_prod = prod_df.select_dtypes(include=np.number)

drift_results = detect_feature_drift(numeric_ref, numeric_prod, psi_threshold=psi_threshold, ks_alpha=ks_alpha)
drift_alerts = st.session_state.alert_mgr.check_drift(drift_results)
summary_df = drift_summary(drift_results)

n_drifted = summary_df["Drifted"].eq("YES").sum()
c1, c2, c3 = st.columns(3)
c1.metric("Features Monitored", len(drift_results))
c2.metric("Drifted Features", int(n_drifted))
c3.metric("Drift Alerts", len(drift_alerts))

# Color-coded table
def color_row(row):
    if row["Severity"] == "high":
        return ["background-color: #ffcccc"] * len(row)
    if row["Severity"] == "medium":
        return ["background-color: #fff3cd"] * len(row)
    return [""] * len(row)

st.dataframe(summary_df.style.apply(color_row, axis=1), use_container_width=True)

# PSI bar chart
psi_vals = [(r.feature, r.statistic) for r in drift_results if r.method == "PSI+KS"]
if psi_vals:
    fig, ax = plt.subplots(figsize=(10, 4))
    features, vals = zip(*psi_vals)
    colors = ["#e74c3c" if v > 0.25 else "#f39c12" if v > 0.1 else "#2ecc71" for v in vals]
    ax.barh(features, vals, color=colors)
    ax.axvline(psi_threshold, color="red", linestyle="--", label=f"Threshold ({psi_threshold})")
    ax.set_xlabel("PSI Score")
    ax.set_title("Population Stability Index (PSI) per Feature")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# Feature distribution comparison
st.subheader("Feature Distribution Comparison")
feat_cols = numeric_ref.columns.tolist()
selected_feat = st.selectbox("Select feature to compare", feat_cols)
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.hist(numeric_ref[selected_feat].dropna(), bins=30, alpha=0.6, label="Reference", color="#3498db", density=True)
ax2.hist(numeric_prod[selected_feat].dropna(), bins=30, alpha=0.6, label="Production", color="#e74c3c", density=True)
ax2.set_xlabel(selected_feat)
ax2.set_ylabel("Density")
ax2.legend()
ax2.set_title(f"Distribution Shift: {selected_feat}")
plt.tight_layout()
st.pyplot(fig2)

# ---- PERFORMANCE MONITORING ----
st.header("Model Performance Over Time")
ref_metrics = compute_classification_metrics(ref_true, ref_pred)
prod_metrics = compute_classification_metrics(prod_true, prod_pred)

m1, m2, m3, m4 = st.columns(4)
for col, (metric, ref_val) in zip([m1, m2, m3, m4], ref_metrics.items()):
    prod_val = prod_metrics.get(metric, 0)
    delta = prod_val - ref_val
    col.metric(metric.upper(), f"{prod_val:.3f}", delta=f"{delta:+.3f}", delta_color="normal")
    st.session_state.alert_mgr.check_performance(prod_val, ref_val, metric)

# Sliding window
windows = sliding_window_metrics(
    np.concatenate([ref_true, prod_true]),
    np.concatenate([ref_pred, prod_pred]),
    task="classification", window_size=window_size, step=window_size // 2
)
metrics_df = metrics_to_df(windows)
deg_df = detect_performance_degradation(metrics_df, perf_metric, threshold=0.03)

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(deg_df["window"], deg_df[perf_metric], marker="o", color="#3498db", linewidth=1.5, label=perf_metric)
ax3.fill_between(deg_df["window"], deg_df[perf_metric], alpha=0.1, color="#3498db")
degraded = deg_df[deg_df["degraded"]]
if not degraded.empty:
    ax3.scatter(degraded["window"], degraded[perf_metric], color="#e74c3c", s=60, zorder=5, label="Degraded")
ax3.axvline(len(windows) * 0.6, color="orange", linestyle="--", alpha=0.7, label="Ref → Prod boundary")
ax3.set_xlabel("Window")
ax3.set_ylabel(perf_metric.upper())
ax3.set_title(f"{perf_metric.upper()} Over Time (sliding window={window_size})")
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)

# ---- ALERTS ----
st.header("Alerts")
summary = st.session_state.alert_mgr.get_summary()
a1, a2, a3, a4 = st.columns(4)
a1.metric("Total Alerts", summary["total"])
a2.metric("Critical", summary["critical"])
a3.metric("Warning", summary["warning"])
a4.metric("Info", summary["info"])

if st.session_state.alert_mgr.alerts:
    alerts_df = pd.DataFrame([{
        "Time": a.timestamp[:19], "Level": a.level.upper(),
        "Category": a.category, "Message": a.message
    } for a in st.session_state.alert_mgr.alerts])

    def color_alert(row):
        if row["Level"] == "CRITICAL":
            return ["background-color: #ffcccc"] * len(row)
        if row["Level"] == "WARNING":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(alerts_df.style.apply(color_alert, axis=1), use_container_width=True)
