"""Case study 1: rediscover Kepler's third law from ~2,900 real exoplanets.

Newtonian prediction (P in years, a in AU, M in solar masses):  P = sqrt(a^3 / M)
Symbolic regression receives only (a, M) and the measured periods, with no physics prior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sr_insights import data, modeling  # noqa: E402

OUT = ROOT / "reports"


def main(n_boot: int = 3) -> dict:
    df = data.load_exoplanets()
    X, y = df[["a_au", "m_star"]], df["period_yr"].to_numpy()
    sr_kwargs = dict(unary_operators=["sqrt", "cube", "square"], maxsize=15, niterations=40)

    # Relative-error weighting: periods span 5 orders of magnitude.
    model = modeling.make_sr(**sr_kwargs)
    model.fit(X, y, weights=1 / y**2)
    eq = str(model.sympy())

    pred_sr = model.predict(X)
    pred_newton = np.sqrt(df["a_au"] ** 3 / df["m_star"])
    rel_err = lambda p: float(np.median(np.abs(p - y) / y))  # noqa: E731

    exps = modeling.power_law_fit(df, "period_yr", ["a_au", "m_star"])
    boot = modeling.bootstrap_equations(X, y, n_boot=n_boot, **sr_kwargs) if n_boot else []

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.loglog(y, pred_sr, ".", ms=3, alpha=0.5)
    lims = [y.min() * 0.5, y.max() * 2]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set(xlabel="observed period [yr]", ylabel="symbolic-regression prediction [yr]", title=f"Discovered: P = {eq}")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "kepler.png", dpi=150); plt.close(fig)

    res = {
        "n_planets": len(df),
        "discovered_equation": eq,
        "median_relative_error_sr": rel_err(pred_sr),
        "median_relative_error_newton": rel_err(pred_newton),
        "exponent_a": exps.loc["a_au"].round(4).to_dict(),
        "exponent_m": exps.loc["m_star"].round(4).to_dict(),
        "bootstrap_equations": boot,
        "pareto_front": modeling.pareto_front(model).to_dict(orient="records"),
    }
    (OUT / "kepler.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "pareto_front"}, indent=2, default=str))
    return res


if __name__ == "__main__":
    main()
