"""Public datasets used in the case studies."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parents[2] / "data"

EXOPLANET_QUERY = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    "select+pl_name,hostname,pl_orbper,pl_orbsmax,st_mass,discoverymethod+from+ps+"
    "where+default_flag=1+and+pl_orbper+is+not+null+and+pl_orbsmax+is+not+null+and+st_mass+is+not+null"
    "&format=csv"
)
CONCRETE_URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"

CONCRETE_COLS = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age_days", "strength_mpa"]


def load_exoplanets() -> pd.DataFrame:
    """NASA Exoplanet Archive: orbital period (years), semi-major axis (AU), host-star mass (solar masses)."""
    path = DATA / "exoplanets.csv"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        path.write_bytes(requests.get(EXOPLANET_QUERY, timeout=120).content)
    df = pd.read_csv(path)
    df = df[(df["pl_orbper"] > 0) & (df["pl_orbsmax"] > 0) & (df["st_mass"] > 0)]
    df = df[df["pl_orbper"] < 1e4]  # drop a handful of very long, poorly constrained orbits
    return df.assign(period_yr=df["pl_orbper"] / 365.25, a_au=df["pl_orbsmax"], m_star=df["st_mass"]).reset_index(drop=True)


def load_concrete() -> pd.DataFrame:
    """UCI Concrete Compressive Strength (Yeh, 1998): 8 mix/age features -> strength in MPa."""
    path = DATA / "Concrete_Data.xls"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(requests.get(CONCRETE_URL, timeout=120).content)) as zf:
            zf.extract("Concrete_Data.xls", DATA)
    df = pd.read_excel(path)
    df.columns = CONCRETE_COLS
    return df
