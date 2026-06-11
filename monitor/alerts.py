"""
Threshold-based alerting for drift and performance degradation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from monitor.drift import DriftResult


@dataclass
class Alert:
    level: str          # "info", "warning", "critical"
    category: str       # "drift", "performance", "data_quality"
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    feature: str | None = None
    metric: str | None = None
    value: float | None = None


class AlertManager:
    def __init__(self):
        self.alerts: list[Alert] = []

    def check_drift(self, drift_results: list[DriftResult]) -> list[Alert]:
        new_alerts = []
        for r in drift_results:
            if r.severity == "high":
                a = Alert(level="critical", category="drift",
                          message=f"Critical drift in '{r.feature}' (PSI/stat={r.statistic:.3f})",
                          feature=r.feature, value=r.statistic)
            elif r.severity == "medium":
                a = Alert(level="warning", category="drift",
                          message=f"Moderate drift in '{r.feature}' (PSI/stat={r.statistic:.3f})",
                          feature=r.feature, value=r.statistic)
            else:
                continue
            new_alerts.append(a)
        self.alerts.extend(new_alerts)
        return new_alerts

    def check_performance(self, current: float, baseline: float, metric: str, warning_drop: float = 0.03, critical_drop: float = 0.07) -> list[Alert]:
        new_alerts = []
        drop = baseline - current
        if drop >= critical_drop:
            a = Alert(level="critical", category="performance",
                      message=f"{metric} dropped by {drop:.3f} from baseline {baseline:.3f} → {current:.3f}",
                      metric=metric, value=drop)
            new_alerts.append(a)
        elif drop >= warning_drop:
            a = Alert(level="warning", category="performance",
                      message=f"{metric} dropped by {drop:.3f} from baseline {baseline:.3f} → {current:.3f}",
                      metric=metric, value=drop)
            new_alerts.append(a)
        self.alerts.extend(new_alerts)
        return new_alerts

    def check_missing_rate(self, df, threshold: float = 0.1) -> list[Alert]:
        new_alerts = []
        for col in df.columns:
            rate = df[col].isna().mean()
            if rate > threshold:
                a = Alert(level="warning", category="data_quality",
                          message=f"High missing rate in '{col}': {rate:.1%}",
                          feature=col, value=rate)
                new_alerts.append(a)
        self.alerts.extend(new_alerts)
        return new_alerts

    def get_summary(self) -> dict:
        return {
            "total": len(self.alerts),
            "critical": sum(1 for a in self.alerts if a.level == "critical"),
            "warning": sum(1 for a in self.alerts if a.level == "warning"),
            "info": sum(1 for a in self.alerts if a.level == "info"),
        }

    def clear(self):
        self.alerts = []
