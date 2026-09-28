"""Symbolic regression helpers, baselines and statistical checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score


def make_sr(**overrides):
    """PySR regressor with a deterministic, reproducible configuration."""
    from pysr import PySRRegressor

    params = dict(
        niterations=60,
        populations=16,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "square", "cube", "log", "exp"],
        maxsize=25,
        model_selection="best",
        parsimony=1e-3,
        deterministic=True,
        parallelism="serial",
        random_state=0,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )
    params.update(overrides)
    return PySRRegressor(**params)


def regression_metrics(y_true, y_pred) -> dict:
    return {"r2": r2_score(y_true, y_pred), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def power_law_fit(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    """OLS on logs: log(y) = c + sum_k alpha_k log(x_k). Returns exponents with 95% CIs (HC3)."""
    X = sm.add_constant(np.log(df[features]))
    fit = sm.OLS(np.log(df[target]), X).fit(cov_type="HC3")
    ci = fit.conf_int()
    return pd.DataFrame({"estimate": fit.params, "ci_low": ci[0], "ci_high": ci[1]})


def pareto_front(model) -> pd.DataFrame:
    """Accuracy/complexity trade-off discovered by PySR."""
    eq = model.equations_
    return eq[["complexity", "loss", "equation"]].copy()


def bootstrap_equations(X: pd.DataFrame, y: np.ndarray, n_boot: int = 5, seed: int = 0, **sr_kwargs) -> list[str]:
    """Refit symbolic regression on bootstrap resamples to check that the discovered form is stable."""
    rng = np.random.default_rng(seed)
    found = []
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        model = make_sr(random_state=b, **sr_kwargs)
        model.fit(X.iloc[idx], y[idx])
        found.append(str(model.sympy()))
    return found
