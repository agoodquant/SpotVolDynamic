"""Raw quotes -> one smile summary per (date, expiry): forward, own implied vols, ATM vol, skew."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import store
from .config import FALLBACK_RATE

MIN_DTE = 5
MAX_REL_SPREAD = 0.6


def black(F, K, T, vol, is_call):
    sd = vol * np.sqrt(T)
    d1 = np.log(F / K) / sd + 0.5 * sd
    call = F * norm.cdf(d1) - K * norm.cdf(d1 - sd)
    return np.where(is_call, call, call - (F - K))


def implied_vol(price, F, K, T, is_call):
    lo = np.full(len(price), 1e-3)
    hi = np.full(len(price), 8.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        below = black(F, K, T, mid, is_call) < price
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    v = 0.5 * (lo + hi)
    v[(v < 0.011) | (v > 7.9)] = np.nan
    return v


def _side(g, cp):
    s = g[g["call_put"] == cp].drop_duplicates("strike").set_index("strike").sort_index()
    mid = (s["bid"] + s["ask"]) / 2
    ok = (s["bid"] > 0) & (s["ask"] >= s["bid"]) & ((s["ask"] - s["bid"]) <= MAX_REL_SPREAD * mid)
    return mid[ok]


def _at_delta(deltas, values, target):
    o = np.argsort(deltas)
    d, v = deltas[o], values[o]
    if len(d) < 2 or target < d[0] or target > d[-1]:
        return np.nan
    return float(np.interp(target, d, v))


def expiry_smile(g, dte):
    """g: raw quotes of one date and expiry."""
    T = dte / 365.0
    c, p = _side(g, "C"), _side(g, "P")
    both = pd.concat([c.rename("c"), p.rename("p")], axis=1, join="inner")
    if len(both) < 3:
        return None
    cp = both["c"] - both["p"]
    k_star = cp.abs().idxmin()
    df_ = np.exp(-FALLBACK_RATE * T)
    F = k_star + cp[k_star] / df_
    if dte >= 60:
        # put-call parity across strikes: C - P = DF * (F - K) gives the market's own discount and forward
        near = cp[np.abs(cp.index.values / F - 1) < 0.15]
        if len(near) >= 5:
            b, a = np.polyfit(near.index.values, near.values, 1)
            if 0.80 < -b < 1.01 and abs(a / -b / F - 1) < 0.05:
                df_, F = -b, a / -b

    puts, calls = p[p.index < F], c[c.index >= F]
    K = np.concatenate([puts.index.values, calls.index.values])
    is_call = np.concatenate([np.zeros(len(puts), bool), np.ones(len(calls), bool)])
    px = np.concatenate([puts.values, calls.values]) / df_
    vol = implied_vol(px, F, K, T, is_call)
    ok = ~np.isnan(vol)
    K, vol, is_call = K[ok], vol[ok], is_call[ok]
    if len(K) < 5 or is_call.all() or not is_call.any():
        return None

    k = np.log(K / F)
    atm = float(np.interp(0.0, k, vol))
    sd = vol * np.sqrt(T)
    delta = norm.cdf(-k / sd + 0.5 * sd) - (~is_call)
    out = {"dte": dte, "F": float(F), "DF": float(df_), "atm": atm, "n_strikes": len(K),
           "strikes": K.tolist(), "vols": vol.tolist()}
    for d in (25, 10):
        out[f"v{d}p"] = _at_delta(delta[~is_call], vol[~is_call], -d / 100)
        out[f"k{d}p"] = _at_delta(delta[~is_call], K[~is_call], -d / 100)
        out[f"v{d}c"] = _at_delta(delta[is_call], vol[is_call], d / 100)
        out[f"k{d}c"] = _at_delta(delta[is_call], K[is_call], d / 100)
    out["rr25"] = out["v25p"] - out["v25c"]                       # put minus call
    out["fly25"] = (out["v25p"] + out["v25c"]) / 2 - atm
    out["slope"] = (out["v25c"] - out["v25p"]) / np.log(out["k25c"] / out["k25p"])   # dvol/dlnK
    return out


def build(source, symbol, rebuild=False):
    """Compute smiles for raw dates not yet in the smiles table. Returns number of dates added."""
    done = set() if rebuild else store.smile_dates(source, symbol)
    todo = [d for d in store.raw_dates(source, symbol) if d not in done]
    if not todo:
        return 0
    raw = store.read_raw(source, symbol, todo)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["expiration"] = pd.to_datetime(raw["expiration"])
    rows = []
    for (date, exp), g in raw.groupby(["date", "expiration"]):
        dte = (exp - date).days
        if dte < MIN_DTE:
            continue
        s = expiry_smile(g, dte)
        if s:
            rows.append({"date": date, "expiration": exp, "spot": g["spot"].iloc[0], **s})
    store.write_smiles(pd.DataFrame(rows).assign(source=source, symbol=symbol) if rows else pd.DataFrame(), source, symbol, replace=rebuild)
    return len(todo)
