# ML Model Monitor

Production-grade ML model monitoring system for detecting **data drift**, **concept drift**, and **performance degradation** with a real-time Streamlit dashboard.

## Features

### Drift Detection
- **PSI (Population Stability Index)** — industry-standard drift metric used in credit/finance ML
- **Kolmogorov-Smirnov test** — statistical test for distribution shift
- **KL Divergence** — information-theoretic drift measure
- **Chi-Square test** — categorical feature drift
- Color-coded severity: none / low / medium / high

### Performance Monitoring
- Sliding window metrics over time (accuracy, F1, precision, recall, AUC)
- Baseline vs production comparison
- Degradation detection with configurable thresholds
- Visual timeline with degradation markers

### Alerting
- `AlertManager` with warning / critical levels
- Checks: drift severity, performance drop, missing data rate
- Alert history with timestamps

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Use as a Library

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

## Tech Stack
`Python` `Scikit-learn` `SciPy` `Pandas` `NumPy` `Streamlit` `Matplotlib`
