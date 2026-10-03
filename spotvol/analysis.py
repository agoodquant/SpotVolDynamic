"""Spot/vol dynamics from the smile store.

A "pair" is two consecutive observation dates and one expiry listed on both. For each pair:
  r     spot return in % (parity forward of that expiry)
  dAtm  change in ATM vol, vol points
  dFix  change in vol at the first day's ATM strike, vol points (zero under sticky strike)
beta is the OLS slope of a vol change on r: vol points per 1% spot move.
"""
import numpy as np
import pandas as pd

from . import store, yahoo
from .config import SOURCES, TENORS

MAX_GAP_DAYS = 5
MAX_ABS_RETURN = 35.0   # % log return between observations; beyond this it is a split, not a move


def load_smiles(symbol):
    sm = store.read_smiles(symbol)
    if sm.empty:
        return None
    sm["date"] = pd.to_datetime(sm["date"])
    sm["expiration"] = pd.to_datetime(sm["expiration"])
    return sm


def tenor_band(tenor_days):
    """Range of days to expiry accepted as representing a tenor."""
    return max(14, 0.65 * tenor_days), 1.6 * tenor_days


def _earn_flag(d0, d1, earn):
    for e, when in zip(earn["date"], earn["when"]):
        if (when == "amc" and d0 <= e < d1) or (when == "bmo" and d0 < e <= d1) or (when == "unknown" and d0 <= e <= d1):
            return True
    return False


def make_pairs(sm, tenor_days, earn):
    lo, hi = tenor_band(tenor_days)
    rows = []
    for src, s in sm.groupby("source"):
        by_date = {d: g.set_index("expiration") for d, g in s.groupby("date")}
        dates = sorted(by_date)
        for d0, d1 in zip(dates[:-1], dates[1:]):
            if (d1 - d0).days > MAX_GAP_DAYS:
                continue
            g0, g1 = by_date[d0], by_date[d1]
            common = [e for e in g1.index if e in g0.index and lo <= g1.at[e, "dte"] <= hi]
            if not common:
                continue
            exp = min(common, key=lambda e: abs(g1.at[e, "dte"] - tenor_days))
            e0, e1 = g0.loc[exp], g1.loc[exp]
            K, v = np.asarray(e1["strikes"]), np.asarray(e1["vols"])
            fix = np.interp(e0["F"], K, v) if K[0] <= e0["F"] <= K[-1] else np.nan
            rows.append({"source": src, "d0": d0, "d1": d1, "dte": e1["dte"],
                         "r": 100 * np.log(e1["F"] / e0["F"]),
                         "dAtm": 100 * (e1["atm"] - e0["atm"]), "dFix": 100 * (fix - e0["atm"]),
                         "atm0": 100 * e0["atm"], "slope0": e0["slope"],
                         "earn": _earn_flag(d0, d1, earn)})
    p = pd.DataFrame(rows)
    if p.empty:
        return p
    p["pref"] = p["source"].map({s: i for i, s in enumerate(SOURCES)})
    p = p.sort_values(["d1", "pref"]).drop_duplicates("d1").drop(columns="pref")
    p = p[~((p["r"] == 0) & (p["dAtm"] == 0))]            # stale vendor duplicates on holidays
    p = p[p["r"].abs() < MAX_ABS_RETURN]                  # stock splits
    return p.dropna(subset=["dFix"]).reset_index(drop=True)


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 8 or np.var(x) == 0:
        return {"n": n, "beta": np.nan, "t": np.nan, "corr": np.nan}
    dx, dy = x - x.mean(), y - y.mean()
    sxx, sxy, syy = dx @ dx, dx @ dy, dy @ dy
    b = sxy / sxx
    se = np.sqrt(max(syy - b * sxy, 0) / (n - 2) / sxx)
    return {"n": n, "beta": b, "t": b / se if se > 0 else np.nan, "corr": sxy / np.sqrt(sxx * syy) if syy > 0 else np.nan}


def summary(p):
    """One row of headline statistics for a set of pairs."""
    a, f = ols(p["r"], p["dAtm"]), ols(p["r"], p["dFix"])
    up, dn = p[p["r"] > 0], p[p["r"] < 0]
    slope = p["slope0"].mean()
    return {"n": len(p), "beta_atm": a["beta"], "t_atm": a["t"], "beta_fix": f["beta"], "t_fix": f["t"],
            "corr_fix": f["corr"], "skew_slope": slope,
            "beta_fix_up": ols(up["r"], up["dFix"])["beta"], "beta_fix_down": ols(dn["r"], dn["dFix"])["beta"]}


def regimes(p):
    last = p["d1"].max()
    rows = [("Full sample", p)]
    for label, days in (("Last 12 months", 365), ("Last 6 months", 182), ("Last 3 months", 91)):
        rows.append((label, p[p["d1"] > last - pd.Timedelta(days=days)]))
    half = p["d1"].dt.year.astype(str) + "H" + ((p["d1"].dt.month + 5) // 6).astype(str)
    rows += [(h, g) for h, g in p.groupby(half)]
    return pd.DataFrame([{"sample": k, **summary(g)} for k, g in rows if len(g) >= 8])


def horizons(p, hs=(1, 5, 10)):
    rows = []
    for h in hs:
        grp = np.arange(len(p)) // h                       # non-overlapping windows of h observations
        agg = p.groupby(grp)[["r", "dAtm", "dFix"]].sum()[p.groupby(grp).size() == h]
        a, f = ols(agg["r"], agg["dAtm"]), ols(agg["r"], agg["dFix"])
        rows.append({"horizon_obs": h, "n": a["n"], "beta_atm": a["beta"], "t_atm": a["t"], "beta_fix": f["beta"], "t_fix": f["t"]})
    return pd.DataFrame(rows)


BUCKETS = [-np.inf, -6, -3, -1, 1, 3, 6, np.inf]
BUCKET_LABELS = ["< -6%", "-6 to -3%", "-3 to -1%", "-1 to 1%", "1 to 3%", "3 to 6%", "> 6%"]


def buckets(p):
    b = pd.cut(p["r"], BUCKETS, labels=BUCKET_LABELS, right=False)
    g = p.groupby(b, observed=False)
    return pd.DataFrame({"n": g.size(), "avg_ret": g["r"].mean(), "avg_dAtm": g["dAtm"].mean(), "avg_dFix": g["dFix"].mean(),
                         "vol_up_freq": g["dFix"].apply(lambda x: (x > 0).mean() if len(x) else np.nan)}).reset_index(names="bucket")


def rolling(p, window=60):
    rows = []
    for i in range(window, len(p) + 1):
        w = p.iloc[i - window:i]
        rows.append({"date": w["d1"].iloc[-1], "beta_fix": ols(w["r"], w["dFix"])["beta"],
                     "beta_atm": ols(w["r"], w["dAtm"])["beta"], "skew_slope": w["slope0"].mean()})
    return pd.DataFrame(rows)


def levels(sm, tenor_days):
    """Per date, the expiry nearest the tenor (one source per date, preferred source first)."""
    lo, hi = tenor_band(tenor_days)
    s = sm[(sm["dte"] >= lo) & (sm["dte"] <= hi)].copy()
    s["dist"] = (s["dte"] - tenor_days).abs()
    s["pref"] = s["source"].map({x: i for i, x in enumerate(SOURCES)})
    s = s.sort_values(["date", "pref", "dist"]).drop_duplicates("date").reset_index(drop=True)
    # back-adjust the forward for splits so the price chart is continuous
    step = np.log(s["F"] / s["F"].shift()).fillna(0)
    jump = step.where(step.abs() >= np.log1p(MAX_ABS_RETURN / 100), 0)
    s["F_adj"] = s["F"] * np.exp(jump[::-1].cumsum()[::-1] - jump)
    return s


def analyse(symbol):
    """Everything the report needs, per tenor that has data."""
    sm = load_smiles(symbol)
    if sm is None:
        return None
    earn = yahoo.earnings(symbol)
    out = {"symbol": symbol, "smiles": sm, "earnings": earn, "tenors": {}}
    for name, days in TENORS.items():
        allp = make_pairs(sm, days, earn)
        if allp.empty:
            continue
        out["tenors"][name] = {"pairs": allp[~allp["earn"]].reset_index(drop=True), "n_earn": int(allp["earn"].sum()),
                               "levels": levels(sm, days)}
    return out
