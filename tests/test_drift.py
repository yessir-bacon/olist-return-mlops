import numpy as np
import pandas as pd

from olist_returns import config
from olist_returns.drift import drift_report, drift_share


def make_data(n=500, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(shift, 1, n) for c in config.NUM_COLS})
    for c in config.CAT_COLS:
        df[c] = rng.choice(["a", "b", "c"], n)
    return df


def test_no_drift_on_identical_data():
    ref = make_data()
    assert drift_share(drift_report(ref, ref.copy())) == 0


def test_detects_shifted_data():
    assert drift_share(drift_report(make_data(), make_data(shift=3, seed=1))) > 0.5
