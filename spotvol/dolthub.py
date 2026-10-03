"""History backfill from the free DoltHub options database (post-no-preference/options).

Covers US optionable names from 2019, end of day, three front expiries only.
Queried one calendar week at a time by exact date, which is the only access pattern the API serves reliably.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from . import store

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
ROW_CAP = 1000
SETTLED_DAYS = 7     # an empty date older than this is recorded and not asked for again


def _query(sql, tries=6):
    for i in range(tries):
        try:
            r = requests.get(API, params={"q": sql}, timeout=120).json()
            if r.get("query_execution_status") == "Success":
                return r["rows"]
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 + 3 * i)
    raise RuntimeError(f"DoltHub query failed: {sql[:120]}")


def _fetch(symbol, dates):
    inlist = ",".join(f"'{d:%Y-%m-%d}'" for d in dates)
    try:
        rows = _query("select date,expiration,strike,call_put,bid,ask,vol,delta from option_chain "
                      f"where act_symbol='{symbol}' and date in ({inlist})", tries=6 if len(dates) == 1 else 2)
    except RuntimeError:
        if len(dates) == 1:
            raise
        rows = [None] * ROW_CAP      # a week that times out is retried one day at a time
    if len(rows) >= ROW_CAP and len(dates) > 1:
        rows = [r for d in dates for r in _fetch(symbol, [d])]
    return rows


def _frame(symbol, dates):
    """Frame of quotes, None if the dates have no data, or the exception if the API kept failing."""
    try:
        rows = _fetch(symbol, dates)
    except RuntimeError as e:
        return e
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    for c in ["strike", "bid", "ask", "vol", "delta"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["call_put"] = df["call_put"].str[0]
    return df.rename(columns={"vol": "src_vol", "delta": "src_delta"})


def backfill(symbol, start):
    empty = store.empty_dates("dolthub", symbol)
    have = set(store.raw_dates("dolthub", symbol))
    today = pd.Timestamp.today().normalize()
    todo = [d for d in pd.bdate_range(start, today) if d not in have and d not in empty]
    weeks = {}
    for d in todo:
        weeks.setdefault(d.to_period("W"), []).append(d)
    chunks = list(weeks.values())
    with ThreadPoolExecutor(3) as ex:
        results = list(ex.map(lambda c: _frame(symbol, c), chunks))
    frames = [f for f in results if isinstance(f, pd.DataFrame)]
    failed = {d for c, f in zip(chunks, results) if isinstance(f, Exception) for d in c}
    got = set(store.write_raw(pd.concat(frames, ignore_index=True), "dolthub", symbol)) if frames else set()
    store.add_empty_dates("dolthub", symbol, [d for d in todo if d not in got and d not in failed and (today - d).days > SETTLED_DAYS])
    return len(got), len(failed)
