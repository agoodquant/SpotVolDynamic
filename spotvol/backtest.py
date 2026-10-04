"""Rolling out-of-sample backtest of every model in spotvol.models.

Each model predicts, one observation at a time, the next change in vol given that observation's spot move, using
coefficients fitted only on earlier observations. Nothing from the future enters a prediction except the size of the
spot move itself, which is what the charge is conditional on.
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import analysis as an
from .config import REPORTS
from .models import MODELS, PRIMARY, EwmModel, design, get, markov, winsorize
from .report import AQUA, BLUE, CSS, MUTED, ORANGE, SURFACE, _style, _table

START = 200       # observations before the first prediction
STEP = 5          # refit interval for models that are not refitted every day
SENSITIVITY_HALFLIVES = (5, 10, 20, 40, 80)


ROBUST_DROP = 3   # robust R-squared leaves out this many of the largest realized vol moves


def _score(y, pred, mask, r, big=1.0):
    q = mask & ~np.isnan(pred)
    if q.sum() < 10:
        return {"n": int(q.sum()), "r2": np.nan, "r2_robust": np.nan, "hit": np.nan, "mae": np.nan}
    e = y[q] - pred[q]
    qb = q & (np.abs(r) >= big)
    keep = np.argsort(np.abs(y[q]))[:-ROBUST_DROP]          # a single extreme day can swing R-squared by several points
    return {"n": int(q.sum()), "r2": 1 - (e ** 2).sum() / (y[q] ** 2).sum(), "mae": np.abs(e).mean(),
            "r2_robust": 1 - (e[keep] ** 2).sum() / (y[q][keep] ** 2).sum(),
            "hit": (np.sign(pred[qb]) == np.sign(y[qb])).mean() if qb.sum() else np.nan}


def _scores(y, pred, test, r):
    return {"all": _score(y, pred, test, r), "up": _score(y, pred, test & (r > 0), r), "down": _score(y, pred, test & (r < 0), r)}


def run(symbol, tenor="1m", target="dFix"):
    p = an.analyse(symbol)["tenors"][tenor]["pairs"]
    y, r, closed = p[target].to_numpy(), p["r"].to_numpy(), p["closed"].to_numpy()
    paths = {m.name: m.walk_forward(y, r, START, STEP, closed) for m in MODELS}
    test = ~np.isnan(paths[PRIMARY]["pred"].to_numpy())
    df = pd.DataFrame({"date": p["d1"].to_numpy(), "r": r, "real": y})
    for name, w in paths.items():
        df[f"pred_{name}"], df[f"b_up_{name}"], df[f"b_dn_{name}"] = w["pred"], w["b_up"], w["b_dn"]
    scores = {m.name: _scores(y, paths[m.name]["pred"].to_numpy(), test, r) for m in MODELS}
    sens = {h: _scores(y, EwmModel(h).walk_forward(y, r, START, closed=closed)["pred"].to_numpy(), test, r) for h in SENSITIVITY_HALFLIVES}
    hindsight = markov.fit(winsorize(y), design(r), 2)["smooth"][:, -1]
    agree = ((paths["markov"]["p_hi"].to_numpy()[test] > 0.5) == (hindsight[test] > 0.5)).mean()
    return {"symbol": symbol, "tenor": tenor, "target": target, "df": df, "test": test, "scores": scores, "sensitivity": sens,
            "y": y, "r": r, "regime_agree": agree}


# ---------- tables ----------

def _pct(x, signed=True):
    return "–" if np.isnan(x) else f"{x:+.1%}" if signed else f"{x:.0%}"


def _accuracy_table(bt):
    mae0 = np.abs(bt["y"][bt["test"]]).mean()
    rows = [{"Model": m.label, "Obs": s["all"]["n"], "R², all days": _pct(s["all"]["r2"]),
             f"R², without {ROBUST_DROP} largest moves": _pct(s["all"]["r2_robust"]), "R², up days": _pct(s["up"]["r2"]),
             "R², down days": _pct(s["down"]["r2"]), "Direction right, up days": _pct(s["up"]["hit"], False),
             "Direction right, down days": _pct(s["down"]["hit"], False), "Mean abs. error": f"{s['all']['mae']:.2f}"}
            for m in MODELS for s in [bt["scores"][m.name]]]
    rows.append({"Model": "Predicting no change", "Obs": rows[0]["Obs"], "R², all days": "0.0%", f"R², without {ROBUST_DROP} largest moves": "0.0%",
                 "R², up days": "0.0%", "R², down days": "0.0%",
                 "Direction right, up days": "–", "Direction right, down days": "–", "Mean abs. error": f"{mae0:.2f}"})
    return pd.DataFrame(rows)


def _sensitivity_table(bt):
    return pd.DataFrame([{"Half-life (obs)": h, "R², all days": _pct(s["all"]["r2"]), "R², up days": _pct(s["up"]["r2"]), "R², down days": _pct(s["down"]["r2"])}
                         for h, s in bt["sensitivity"].items()])


def _sum_table(d, key, key_label):
    """Realized against predicted totals, for up days and down days, grouped by `key`."""
    rows = []
    for k, g in d.groupby(key, observed=True, sort=False):
        up, dn = g[g["r"] > 0], g[g["r"] < 0]
        rows.append({key_label: str(k), "Up days": len(up), "Realized": f"{up['real'].sum():+.0f}",
                     "Short memory": f"{up[f'pred_{PRIMARY}'].sum():+.0f}", "Markov": f"{up['pred_markov'].sum():+.0f}",
                     "Down days": len(dn), "Realized ": f"{dn['real'].sum():+.0f}",
                     "Short memory ": f"{dn[f'pred_{PRIMARY}'].sum():+.0f}", "Markov ": f"{dn['pred_markov'].sum():+.0f}"})
    return pd.DataFrame(rows)


def _move_table(d):
    b = pd.cut(d["r"], an.BUCKETS, labels=an.BUCKET_LABELS, right=False)
    rows = [{"Spot move": str(k), "Obs": len(g), "Avg realized Δvol": f"{g['real'].mean():+.2f}",
             "Avg predicted, short memory": f"{g[f'pred_{PRIMARY}'].mean():+.2f}", "Avg predicted, Markov": f"{g['pred_markov'].mean():+.2f}",
             "Vol up": f"{(g['real'] > 0).mean():.0%}"} for k, g in d.groupby(b, observed=True)]
    return pd.DataFrame(rows)


# ---------- charts ----------

def _fig_cumulative(d):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Up days: cumulative Δvol, vol pts", "Down days: cumulative Δvol, vol pts"))
    series = (("real", "Realized", BLUE), (f"pred_{PRIMARY}", "Predicted, short memory", ORANGE), ("pred_markov", "Predicted, Markov", AQUA))
    for col, q in ((1, d["r"] > 0), (2, d["r"] < 0)):
        g = d[q]
        for key, name, color in series:
            fig.add_scatter(x=g["date"], y=g[key].cumsum(), line=dict(color=color, width=2), name=name, legendgroup=key, showlegend=col == 1, row=1, col=col)
    return _style(fig, 400).update_layout(legend=dict(y=1.18))


def _fig_betas(d):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("β up forecast", "β down forecast"))
    for col, side in ((1, "b_up"), (2, "b_dn")):
        for name, label, color in ((PRIMARY, "Short memory", ORANGE), ("markov", "Markov", AQUA)):
            fig.add_scatter(x=d["date"], y=d[f"{side}_{name}"], line=dict(color=color, width=2), name=label, legendgroup=name, showlegend=col == 1, row=1, col=col)
    fig.update_yaxes(title="vol pts per 1% spot move", col=1)
    return _style(fig, 360).update_layout(legend=dict(y=1.2))


def _fig_calibration(d):
    key = f"pred_{PRIMARY}"
    g = d.assign(bin=pd.qcut(d[key], 8, duplicates="drop")).groupby("bin", observed=True).agg(pred=(key, "mean"), real=("real", "mean"), n=("real", "size"))
    lim = [min(g["pred"].min(), g["real"].min()) - 0.2, max(g["pred"].max(), g["real"].max()) + 0.2]
    fig = go.Figure()
    fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color=MUTED, width=1), name="Perfect calibration", hoverinfo="skip")
    fig.add_scatter(x=g["pred"], y=g["real"], mode="markers", marker=dict(color=ORANGE, size=11, line=dict(color=SURFACE, width=2)), name="Eighth of predictions",
                    customdata=g["n"], hovertemplate="predicted %{x:+.2f}<br>realized %{y:+.2f}<br>%{customdata} obs<extra></extra>")
    fig.update_xaxes(title="Average predicted Δvol, vol pts")
    fig.update_yaxes(title="Average realized Δvol, vol pts")
    return _style(fig, 400).update_layout(hovermode="closest")


# ---------- report ----------

def write_backtest(symbol):
    bt = run(symbol)
    d = bt["df"][bt["test"]].reset_index(drop=True)
    s = bt["scores"][PRIMARY]
    figs = [_fig_cumulative(d), _fig_calibration(d), _fig_betas(d)]
    divs = [f.to_html(full_html=False, include_plotlyjs="inline" if i == 0 else False, config={"displaylogo": False}) for i, f in enumerate(figs)]
    quarter = d["date"].dt.to_period("Q").astype(str)
    parts = [f"<h1>{symbol} model backtest</h1>",
             f"<p>{len(d)} out-of-sample predictions from {d['date'].iloc[0]:%Y-%m-%d} to {d['date'].iloc[-1]:%Y-%m-%d}, tenor {bt['tenor']}, earnings excluded. "
             "Each model predicts the next change in fixed-strike vol given the spot move, using only earlier data.</p>",
             f'<p class="verdict">{html.escape(get(PRIMARY).label)}: on up days it explains {s["up"]["r2"]:.0%} of the vol-change variance and gets the direction right '
             f'{s["up"]["hit"]:.0%} of the time (moves of 1% or more); on down days {s["down"]["r2"]:.0%} and {s["down"]["hit"]:.0%}.</p>',
             "<h2>Accuracy by model</h2><p>R² is against predicting no vol change; negative means worse than predicting nothing. "
             f"The second R² leaves out the {ROBUST_DROP} days with the largest realized vol moves, since one extreme day can swing the first by several points. "
             "Direction right counts days with a spot move of at least 1%.</p>", _table(_accuracy_table(bt)),
             "<h2>What each model would have charged against what happened</h2>"
             "<p>Running totals of predicted and realized fixed-strike vol change, per unit of vega, on up days and down days separately.</p>",
             f'<div class="card">{divs[0]}</div>', _table(_sum_table(d.assign(q=quarter), "q", "Quarter")),
             "<h2>By size of spot move</h2>", _table(_move_table(d)),
             f"<h2>Calibration, {html.escape(get(PRIMARY).label.lower())}</h2><p>Predictions sorted into eight equal groups: when the model predicts more, does more happen?</p>",
             f'<div class="card">{divs[1]}</div>',
             "<h2>The β forecasts as they stood each day</h2>", f'<div class="card">{divs[2]}</div>',
             f"<p>The Markov regime called in real time matches the one identified with hindsight (full-sample fit) on {bt['regime_agree']:.0%} of days.</p>",
             "<h2>How much memory?</h2><p>The short-memory model at different half-lives. A shorter half-life turns faster but is noisier.</p>", _table(_sensitivity_table(bt))]
    REPORTS.mkdir(exist_ok=True)
    d.to_csv(REPORTS / f"{symbol}_backtest.csv", index=False)
    path = REPORTS / f"{symbol}_backtest.html"
    path.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                    f"<title>{html.escape(symbol)} backtest</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>", encoding="utf-8")
    return path
