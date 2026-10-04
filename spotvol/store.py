"""DuckDB store: one file, data/spotvol.duckdb.

  raw_quotes   every quote as collected; a (source, symbol, date) slice is written once and never overwritten
  smiles       one row per (date, source, symbol, expiration), derived from raw_quotes
  earnings     cached earnings calendar
  beta_monitor one row per (date, symbol, tenor, model): the next-day betas as each model estimated them on that date
  empty_dates  dates a source was asked for and had nothing, so they are not asked for again
"""
import duckdb
import pandas as pd

from .config import DB

RAW_COLS = {"date": "DATE", "snap_ts": "TIMESTAMP", "source": "VARCHAR", "symbol": "VARCHAR", "expiration": "DATE",
            "strike": "DOUBLE", "call_put": "VARCHAR", "bid": "DOUBLE", "ask": "DOUBLE", "last": "DOUBLE",
            "volume": "DOUBLE", "open_interest": "DOUBLE", "src_vol": "DOUBLE", "src_delta": "DOUBLE", "spot": "DOUBLE"}
SMILE_COLS = {"date": "DATE", "source": "VARCHAR", "symbol": "VARCHAR", "expiration": "DATE", "spot": "DOUBLE",
              "dte": "INTEGER", "F": "DOUBLE", "DF": "DOUBLE", "atm": "DOUBLE", "n_strikes": "INTEGER",
              "strikes": "DOUBLE[]", "vols": "DOUBLE[]",
              "v25p": "DOUBLE", "k25p": "DOUBLE", "v25c": "DOUBLE", "k25c": "DOUBLE",
              "v10p": "DOUBLE", "k10p": "DOUBLE", "v10c": "DOUBLE", "k10c": "DOUBLE",
              "rr25": "DOUBLE", "fly25": "DOUBLE", "slope": "DOUBLE"}
MONITOR_COLS = {"date": "DATE", "symbol": "VARCHAR", "tenor": "VARCHAR", "target": "VARCHAR", "model": "VARCHAR",
                "a_next": "DOUBLE", "b_up_next": "DOUBLE", "b_dn_next": "DOUBLE", "resid_sd": "DOUBLE",
                "n_regimes": "INTEGER", "regime": "VARCHAR", "prob": "DOUBLE", "halflife": "INTEGER", "c_closed": "DOUBLE",
                "crash_coef": "DOUBLE", "crash_threshold": "DOUBLE"}


def _ddl(name, cols):
    return f"create table if not exists {name} (" + ", ".join(f'"{c}" {t}' for c, t in cols.items()) + ")"


def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    con.execute(_ddl("raw_quotes", RAW_COLS))
    con.execute(_ddl("smiles", SMILE_COLS))
    con.execute('create table if not exists earnings (symbol VARCHAR, date DATE, "when" VARCHAR, fetched_at TIMESTAMP)')
    con.execute("create table if not exists empty_dates (source VARCHAR, symbol VARCHAR, date DATE)")
    con.execute(_ddl("beta_monitor", MONITOR_COLS))
    for col in ("c_closed", "crash_coef", "crash_threshold"):
        con.execute(f"alter table beta_monitor add column if not exists {col} DOUBLE")
    return con


def _insert(con, table, cols, df):
    df = df.reindex(columns=list(cols))
    select = ", ".join(f'cast("{c}" as {t}) as "{c}"' for c, t in cols.items())
    con.register("_df", df)
    con.execute(f"insert into {table} by name select {select} from _df")
    con.unregister("_df")


def raw_dates(source, symbol):
    with connect() as con:
        rows = con.execute("select distinct date from raw_quotes where source=? and symbol=? order by 1", [source, symbol]).fetchall()
    return [pd.Timestamp(r[0]) for r in rows]


def has_raw(source, symbol, date):
    with connect() as con:
        return con.execute("select count(*) from (select 1 from raw_quotes where source=? and symbol=? and date=? limit 1)",
                           [source, symbol, pd.Timestamp(date).date()]).fetchone()[0] > 0


def write_raw(df, source, symbol):
    """df may hold several dates; dates already stored for this source and symbol are skipped. Returns dates written."""
    df = df.assign(source=source, symbol=symbol, date=pd.to_datetime(df["date"]).dt.normalize())
    have = set(raw_dates(source, symbol))
    df = df[~df["date"].isin(have)]
    if df.empty:
        return []
    with connect() as con:
        _insert(con, "raw_quotes", RAW_COLS, df)
    return sorted(df["date"].unique())


def read_raw(source, symbol, dates):
    with connect() as con:
        return con.execute("select * from raw_quotes where source=? and symbol=? and date in (select unnest(?))",
                           [source, symbol, [pd.Timestamp(d).date() for d in dates]]).df()


def smile_dates(source, symbol):
    with connect() as con:
        rows = con.execute("select distinct date from smiles where source=? and symbol=?", [source, symbol]).fetchall()
    return {pd.Timestamp(r[0]) for r in rows}


def read_smiles(symbol):
    with connect() as con:
        return con.execute("select * from smiles where symbol=? order by date, expiration", [symbol]).df()


def write_smiles(df, source, symbol, replace=False):
    with connect() as con:
        if replace:
            con.execute("delete from smiles where source=? and symbol=?", [source, symbol])
        if len(df):
            _insert(con, "smiles", SMILE_COLS, df)


def read_earnings(symbol):
    """(frame with date and when, age in days) or (None, None) if never fetched."""
    with connect() as con:
        df = con.execute('select date, "when", fetched_at from earnings where symbol=? order by date', [symbol]).df()
    if df.empty:
        return None, None
    age = (pd.Timestamp.now() - df["fetched_at"].max()).total_seconds() / 86400
    return df[["date", "when"]].dropna(subset=["date"]), age


def write_earnings(symbol, df):
    # a name with no calendar (an ETF) still gets one null row so the fetch is not repeated every run
    rows = df if len(df) else pd.DataFrame({"date": [pd.NaT], "when": [None]})
    rows = rows.assign(symbol=symbol, fetched_at=pd.Timestamp.now())
    with connect() as con:
        con.execute("delete from earnings where symbol=?", [symbol])
        con.register("_e", rows)
        con.execute('insert into earnings by name select symbol, cast(date as DATE) as date, cast("when" as VARCHAR) as "when", fetched_at from _e')


def empty_dates(source, symbol):
    with connect() as con:
        rows = con.execute("select date from empty_dates where source=? and symbol=?", [source, symbol]).fetchall()
    return {pd.Timestamp(r[0]) for r in rows}


def add_empty_dates(source, symbol, dates):
    if not dates:
        return
    df = pd.DataFrame({"source": source, "symbol": symbol, "date": [pd.Timestamp(d).date() for d in dates]})
    with connect() as con:
        con.register("_d", df)
        con.execute("insert into empty_dates select source, symbol, cast(date as DATE) from _d")


def write_monitor(rows):
    """Keeps each model's estimate as it stood on each date, so its track record can be reviewed later."""
    df = pd.DataFrame(rows)
    with connect() as con:
        for _, r in df.iterrows():
            con.execute("delete from beta_monitor where date=? and symbol=? and tenor=? and target=? and model=?",
                        [pd.Timestamp(r["date"]).date(), r["symbol"], r["tenor"], r["target"], r["model"]])
        _insert(con, "beta_monitor", MONITOR_COLS, df)
