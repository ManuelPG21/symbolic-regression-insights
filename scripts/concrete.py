"""Case study 2: interpretable model of concrete compressive strength vs black-box baselines.

Question for a decision-maker: how much accuracy do we give up by using a formula an engineer
can read, audit and put in a spreadsheet, instead of a gradient-boosting model?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sr_insights import data, modeling  # noqa: E402

OUT = ROOT / "reports"


def main() -> dict:
    df = data.load_concrete()
    X, y = df.drop(columns="strength_mpa"), df["strength_mpa"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    lin = LinearRegression().fit(X_tr, y_tr)
    results["linear_regression"] = modeling.regression_metrics(y_te, lin.predict(X_te)) | {"parameters": X.shape[1] + 1}

    gbm = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, random_state=0, verbose=-1).fit(X_tr, y_tr)
    results["lightgbm"] = modeling.regression_metrics(y_te, gbm.predict(X_te)) | {"parameters": "600 trees"}

    sr = modeling.make_sr(niterations=120, maxsize=30, unary_operators=["sqrt", "log", "square"])
    sr.fit(X_tr, y_tr)
    results["symbolic_regression"] = modeling.regression_metrics(y_te, sr.predict(X_te)) | {"equation": str(sr.sympy())}

    # Accuracy vs complexity along the Pareto front, evaluated on held-out data.
    front = modeling.pareto_front(sr)
    front["test_r2"] = [modeling.regression_metrics(y_te, sr.predict(X_te, index=i))["r2"] for i in range(len(front))]
    front.to_csv(OUT / "concrete_pareto_front.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(front["complexity"], front["test_r2"], "o-", label="symbolic regression (test)")
    ax.axhline(results["lightgbm"]["r2"], c="tab:red", ls="--", label="LightGBM (test)")
    ax.axhline(results["linear_regression"]["r2"], c="gray", ls=":", label="linear regression (test)")
    ax.set(xlabel="equation complexity (nodes)", ylabel="test R²", title="Accuracy vs interpretability", ylim=(0, 1))
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "figures" / "concrete_front.png", dpi=150); plt.close(fig)

    # How strength develops with age for a reference mix, according to each model.
    ref = X_tr.median().to_frame().T
    ages = np.array([1, 3, 7, 14, 28, 56, 90, 180, 365])
    grid = pd.concat([ref] * len(ages), ignore_index=True).assign(age_days=ages)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ages, sr.predict(grid), "o-", label="symbolic regression")
    ax.plot(ages, gbm.predict(grid), "s--", label="LightGBM")
    ax.set(xscale="log", xlabel="age [days]", ylabel="predicted strength [MPa]", title="Strength gain with curing age (median mix)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures" / "concrete_age.png", dpi=150); plt.close(fig)

    (OUT / "concrete.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(front[["complexity", "test_r2", "equation"]].to_string())
    return results


if __name__ == "__main__":
    main()
