import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from olist_returns import config


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Keep model features, with categoricals as plain strings for Evidently."""
    return df[config.FEATURES].astype({c: str for c in config.CAT_COLS})


def drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    report = Report([DataDriftPreset()])
    return report.run(current_data=_prep(current), reference_data=_prep(reference))


def drift_share(snapshot) -> float:
    """Share of features flagged as drifted."""
    metrics = snapshot.dict()["metrics"]

    # Preferred: Evidently's summary metric
    for m in metrics:
        if m["metric_name"].startswith("DriftedColumnsCount"):
            return float(m["value"]["share"])

    # Fallback: compute it from the per-column drift tests
    drifted = []
    for m in metrics:
        if m["metric_name"].startswith("ValueDrift"):
            cfg, value = m["config"], float(m["value"])
            if "p_value" in cfg["method"]:
                drifted.append(value < cfg["threshold"])
            else:  # distance-based methods
                drifted.append(value >= cfg["threshold"])
    if not drifted:
        raise ValueError("No drift metrics found in report")
    return sum(drifted) / len(drifted)
