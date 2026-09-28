import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sr_insights import modeling  # noqa: E402


def test_power_law_fit_recovers_exponents():
    rng = np.random.default_rng(0)
    a = rng.lognormal(0, 1, 500)
    m = rng.lognormal(0, 0.3, 500)
    df = pd.DataFrame({"a": a, "m": m, "p": np.sqrt(a**3 / m) * rng.lognormal(0, 0.01, 500)})
    est = modeling.power_law_fit(df, "p", ["a", "m"])
    assert abs(est.loc["a", "estimate"] - 1.5) < 0.01
    assert abs(est.loc["m", "estimate"] + 0.5) < 0.02
    assert est.loc["a", "ci_low"] < 1.5 < est.loc["a", "ci_high"]


def test_regression_metrics_perfect_fit():
    y = np.array([1.0, 2.0, 3.0])
    m = modeling.regression_metrics(y, y)
    assert m["r2"] == 1.0 and m["rmse"] == 0.0
