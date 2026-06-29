# ML Model Monitor

A dashboard for catching data/concept drift on a deployed model before it silently degrades. Built this to understand how drift detection actually works in practice, not just in theory.

## Features

### Drift detection
- PSI (Population Stability Index) — the metric most commonly used in finance/credit ML
- Kolmogorov-Smirnov test for distribution shift
- KL Divergence
- Chi-Square test for categorical feature drift
- Severity buckets: none / low / medium / high

### Performance monitoring
- Sliding-window metrics over time (accuracy, F1, precision, recall, AUC)
- Baseline vs. production comparison
- Degradation flagging with configurable thresholds
- Timeline view with degradation markers

### Alerting
- `AlertManager` with warning/critical levels
- Checks drift severity, performance drop, and missing-data rate
- Keeps an alert history with timestamps

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Using it as a library

```python
import pandas as pd
from monitor.drift import detect_feature_drift, drift_summary
from monitor.alerts import AlertManager

ref = pd.read_csv("reference.csv")
prod = pd.read_csv("production.csv")

results = detect_feature_drift(ref, prod, psi_threshold=0.2)
print(drift_summary(results))

mgr = AlertManager()
mgr.check_drift(results)
print(mgr.get_summary())
```

## Stack
Python, Scikit-learn, SciPy, Pandas, NumPy, Streamlit, Matplotlib

## Notes / things I'd improve
- Thresholds (PSI 0.2, etc.) are the commonly cited defaults, not tuned against a real production dataset
- No persistence layer — everything runs in-memory per session, would need a DB for actual long-term monitoring
- Alerting is just in-app, no integration with Slack/email/PagerDuty etc.
