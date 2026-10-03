"""Daily snapshot of the full Yahoo option chain (all expiries, including LEAPS), plus earnings dates.

Yahoo has no option history: a snapshot not taken is lost. The snapshot is filed under the date of the
latest session with trades, so a weekend or holiday run stores the last close once and then no-ops.
"""
import time

import pandas as pd
import yfinance as yf

from . import store

ET = "America/New_York"


def snapshot(symbol):
    t = yf.Ticker(symbol)
    spot = float(t.fast_info["lastPrice"])
    parts = []
    for exp in t.options:
        ch = t.option_chain(exp)
        for cp, df in (("C", ch.calls), ("P", ch.puts)):
            if len(df):
                parts.append(df.assign(call_put=cp, expiration=pd.Timestamp(exp)))
        time.sleep(0.15)
    if not parts:
        return None, "no chain"
    q = pd.concat(parts, ignore_index=True)
    session = q["lastTradeDate"].max().tz_convert(ET).normalize().tz_localize(None)
    if store.has_raw("yahoo", symbol, session):
        return session, "already stored"
    near = q[(q["strike"] / spot - 1).abs() < 0.1]
    if len(near) == 0 or (near["bid"] > 0).mean() < 0.5:
        return session, "skipped: quotes look stale (most near-the-money bids are zero)"
    out = pd.DataFrame({
        "date": session, "snap_ts": pd.Timestamp.now(tz="UTC").tz_localize(None),
        "expiration": q["expiration"], "strike": q["strike"].astype(float), "call_put": q["call_put"],
        "bid": q["bid"].astype(float), "ask": q["ask"].astype(float), "last": q["lastPrice"].astype(float),
        "volume": q["volume"].astype(float), "open_interest": q["openInterest"].astype(float),
        "src_vol": q["impliedVolatility"].astype(float), "spot": spot,
    })
    store.write_raw(out, "yahoo", symbol)
    return session, f"stored {len(out)} quotes, {q['expiration'].nunique()} expiries"


def earnings(symbol, max_age_days=7):
    """Earnings events as a frame with columns date, when ('amc', 'bmo' or 'unknown'). Empty for ETFs."""
    cached, age = store.read_earnings(symbol)
    if cached is not None and age < max_age_days:
        return cached
    try:
        e = yf.Ticker(symbol).get_earnings_dates(limit=40)
        idx = e.index.tz_convert(ET)
        when = ["amc" if h >= 16 else "bmo" if h <= 9 else "unknown" for h in idx.hour]
        out = pd.DataFrame({"date": idx.tz_localize(None).normalize(), "when": when})
    except Exception:
        if cached is not None:
            return cached
        out = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"), "when": pd.Series([], dtype="object")})
    store.write_earnings(symbol, out)
    return out
