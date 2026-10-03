"""Rolling out-of-sample backtest of the regime-switching beta model.

At each refit date the model sees only observations before that date. It then predicts, one observation at a time,
the next change in fixed-strike vol given that observation's spot move, updating the regime probability with each
new observation but keeping the fitted parameters until the next refit. Nothing from the future enters a prediction
except the size of the spot move itself, which is what the charge is conditional on.
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import analysis as an
from . import regime
from .config import REPORTS
from .report import BLUE, CSS, MUTED, ORANGE, _style, _table

K = 2
START = 200       # observations before the first prediction
STEP = 5          # refit every STEP observations
WINDOW = 250      # length of the rolling-window variant


def predict(y, X, window=None, start=START, step=STEP):
    """Out-of-sample predictions. window=None uses all history to date; otherwise the last `window` observations."""
    T = len(y)
    out = {k: np.full(T, np.nan) for k in ("pred", "b_up", "b_dn", "p_hi")}
    f = None
    for t0 in range(start, T, step):
        lo = 0 if window is None else max(0, t0 - window)
        t1 = min(t0 + step, T)
        yw = regime.winsorize(y[lo:t0])
        f = regime.fit(yw, X[lo:t0], K, init=(f["theta"], f["sig"], f["P"], f["pi"]) if f else None, restarts=6 if f is None else 2)
        yf = np.clip(y[lo:t1], yw.min(), yw.max())
        alpha, _ = regime._filter(regime._emission(yf, X[lo:t1], f["theta"], f["sig"]), f["P"], f["pi"])
        for t in range(t0, t1):
            p = alpha[t - 1 - lo] @ f["P"]
            out["pred"][t] = p @ (f["theta"] @ X[t])
            out["b_up"][t], out["b_dn"][t], out["p_hi"][t] = p @ f["theta"][:, 1], p @ f["theta"][:, 2], p[-1]
    return out


def benchmarks(y, X, start=START, step=STEP):
    T = len(y)
    single, roll = np.full(T, np.nan), np.full(T, np.nan)
    for t0 in range(start, T, step):
        t1 = min(t0 + step, T)
        yw = regime.winsorize(y[:t0])
        single[t0:t1] = X[t0:t1] @ np.linalg.lstsq(X[:t0], yw, rcond=None)[0]
        roll[t0:t1] = X[t0:t1] @ np.linalg.lstsq(X[t0 - 60:t0], yw[-60:], rcond=None)[0]
    return single, roll


def _score(y, pred, mask, r, big=1.0):
    q = mask & ~np.isnan(pred)
    if q.sum() < 10:
        return {"n": int(q.sum())}
    e = y[q] - pred[q]
    qb = q & (np.abs(r) >= big)
    return {"n": int(q.sum()), "r2": 1 - (e ** 2).sum() / (y[q] ** 2).sum(),
            "mae": np.abs(e).mean(), "mae_zero": np.abs(y[q]).mean(),
            "avg_real": y[q].mean(), "avg_pred": pred[q].mean(),
            "hit": (np.sign(pred[qb]) == np.sign(y[qb])).mean() if qb.sum() else np.nan,
            "corr": np.corrcoef(pred[q], y[q])[0, 1] if np.std(pred[q]) > 0 else np.nan}


def run(symbol, tenor="1m", target="dFix"):
    res = an.analyse(symbol)
    p = res["tenors"][tenor]["pairs"]
    y, X, r = p[target].to_numpy(), regime.design(p["r"]), p["r"].to_numpy()
    exp, rol = predict(y, X), predict(y, X, window=WINDOW)
    single, roll60 = benchmarks(y, X)
    full = regime.fit(regime.winsorize(y), X, K)
    df = pd.DataFrame({"date": p["d1"], "r": r, "real": y, "pred": exp["pred"], "pred_roll": rol["pred"], "pred_single": single, "pred_ols60": roll60,
                       "b_up": exp["b_up"], "b_dn": exp["b_dn"], "p_hi": exp["p_hi"], "p_hi_hindsight": full["smooth"][:, -1]})
    test = ~np.isnan(exp["pred"])
    models = {"Regime model, all history to date": exp["pred"], f"Regime model, last {WINDOW} observations": rol["pred"],
              "Single regime, all history to date": single, "Rolling 60 observations, no regimes": roll60}
    subsets = {"all": test, "up": test & (r > 0), "down": test & (r < 0), "up3": test & (r >= 3), "down3": test & (r <= -3)}
    scores = {m: {k: _score(y, v, q, r) for k, q in subsets.items()} for m, v in models.items()}
    return {"symbol": symbol, "tenor": tenor, "df": df, "test": test, "scores": scores, "models": models, "y": y, "r": r,
            "regime_agree": ((exp["p_hi"][test] > 0.5) == (full["smooth"][test, -1] > 0.5)).mean()}


# ---------- report ----------

def _accuracy_table(bt):
    rows = []
    for m, s in bt["scores"].items():
        rows.append({"model": m, "n": s["all"]["n"], "oos_r2": s["all"]["r2"], "oos_r2_up": s["up"]["r2"], "oos_r2_down": s["down"]["r2"],
                     "Direction right, up days": f'{s["up"]["hit"]:.0%}', "Direction right, down days": f'{s["down"]["hit"]:.0%}',
                     "Mean abs. error": f'{s["all"]["mae"]:.2f}', "Mean abs. error, no model": f'{s["all"]["mae_zero"]:.2f}'})
    return pd.DataFrame(rows)


def _bias_table(bt, pred):
    y, r, test = bt["y"], bt["r"], bt["test"]
    rows = []
    for (lo, hi), lab in zip(zip(an.BUCKETS[:-1], an.BUCKETS[1:]), an.BUCKET_LABELS):
        q = test & (r >= lo) & (r < hi)
        if q.sum():
            up = (y[q] > 0).mean()
            rows.append({"bucket": lab, "n": int(q.sum()), "Avg realized Δvol": f"{y[q].mean():+.2f}", "Avg predicted Δvol": f"{pred[q].mean():+.2f}",
                         "Realized − predicted": f"{y[q].mean() - pred[q].mean():+.2f}", "vol_up_freq": up})
    return pd.DataFrame(rows)


def _period_table(bt, pred):
    d = bt["df"]
    half = d["date"].dt.year.astype(str) + "H" + ((d["date"].dt.month + 5) // 6).astype(str)
    rows = []
    for h in sorted(half[bt["test"]].unique()):
        q = bt["test"] & (half == h).to_numpy()
        s_all, s_up, s_dn = (_score(bt["y"], pred, q & m, bt["r"]) for m in (np.ones(len(q), bool), bt["r"] > 0, bt["r"] < 0))
        rows.append({"sample": h, "n": s_all["n"], "oos_r2": s_all.get("r2", np.nan), "oos_r2_up": s_up.get("r2", np.nan), "oos_r2_down": s_dn.get("r2", np.nan),
                     "Avg β up forecast": f'{d.loc[q, "b_up"].mean():+.2f}', "Avg β down forecast": f'{d.loc[q, "b_dn"].mean():+.2f}'})
    return pd.DataFrame(rows)


def _fig_betas(d):
    fig = go.Figure()
    fig.add_scatter(x=d["date"], y=d["b_up"], line=dict(color=BLUE, width=2), name="β up forecast")
    fig.add_scatter(x=d["date"], y=d["b_dn"], line=dict(color=ORANGE, width=2), name="β down forecast")
    fig.update_yaxes(title="vol pts per 1% spot move")
    return _style(fig, 340)


def _fig_cumulative(d):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Up days: cumulative Δvol, vol pts", "Down days: cumulative Δvol, vol pts"))
    for col, q in ((1, d["r"] > 0), (2, d["r"] < 0)):
        g = d[q]
        fig.add_scatter(x=g["date"], y=g["real"].cumsum(), line=dict(color=BLUE, width=2), name="Realized", legendgroup="r", showlegend=col == 1, row=1, col=col)
        fig.add_scatter(x=g["date"], y=g["pred"].cumsum(), line=dict(color=ORANGE, width=2), name="Predicted (what the model would have charged)",
                        legendgroup="p", showlegend=col == 1, row=1, col=col)
    return _style(fig, 380).update_layout(legend=dict(y=1.16))


def _fig_calibration(d):
    g = d.assign(bin=pd.qcut(d["pred"], 8, duplicates="drop")).groupby("bin", observed=True).agg(pred=("pred", "mean"), real=("real", "mean"), n=("real", "size"))
    lim = [min(g["pred"].min(), g["real"].min()) - 0.2, max(g["pred"].max(), g["real"].max()) + 0.2]
    fig = go.Figure()
    fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color=MUTED, width=1), name="Perfect calibration", hoverinfo="skip")
    fig.add_scatter(x=g["pred"], y=g["real"], mode="markers", marker=dict(color=BLUE, size=11, line=dict(color="#fcfcfb", width=2)), name="Octile of predictions",
                    customdata=g["n"], hovertemplate="predicted %{x:+.2f}<br>realized %{y:+.2f}<br>%{customdata} obs<extra></extra>")
    fig.update_xaxes(title="Average predicted Δvol, vol pts")
    fig.update_yaxes(title="Average realized Δvol, vol pts")
    return _style(fig, 400).update_layout(hovermode="closest")


def write_backtest(symbol):
    bt = run(symbol)
    d = bt["df"][bt["test"]].reset_index(drop=True)
    main = bt["models"]["Regime model, all history to date"]
    s = bt["scores"]["Regime model, all history to date"]
    figs = [_fig_cumulative(d), _fig_calibration(d), _fig_betas(d)]
    divs = [f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False, config={"displaylogo": False}) for i, f in enumerate(figs)]
    parts = [f"<h1>{symbol} regime model backtest</h1>",
             f"<p>{len(d)} out-of-sample predictions from {d['date'].iloc[0]:%Y-%m-%d} to {d['date'].iloc[-1]:%Y-%m-%d}, tenor {bt['tenor']}, earnings excluded. "
             f"The model is refitted every {STEP} observations using only earlier data, then predicts the next change in fixed-strike vol given the spot move.</p>",
             f'<p class="verdict">On up days the model explains {s["up"]["r2"]:.0%} of the vol-change variance and gets the direction right {s["up"]["hit"]:.0%} of the time '
             f'(moves of 1% or more); on down days {s["down"]["r2"]:.0%} and {s["down"]["hit"]:.0%}.</p>',
             "<h2>Accuracy by model</h2><p>R² is against predicting no vol change; negative means worse than predicting nothing. "
             "Direction right counts days with a spot move of at least 1%.</p>", _table(_accuracy_table(bt)),
             "<h2>What the model would have charged against what happened</h2>"
             "<p>Running totals of predicted and realized fixed-strike vol change, per unit of vega, on up days and down days separately.</p>",
             f'<div class="card">{divs[0]}</div>',
             "<h2>By size of spot move</h2>", _table(_bias_table(bt, main)),
             "<h2>Calibration</h2><p>Predictions sorted into eight equal groups: when the model predicts more, does more happen?</p>", f'<div class="card">{divs[1]}</div>',
             "<h2>Stability over time</h2>", _table(_period_table(bt, main)),
             "<h2>The β forecasts as they stood each day</h2>", f'<div class="card">{divs[2]}</div>',
             f"<p>The regime called in real time matches the one identified with hindsight (full-sample fit) on {bt['regime_agree']:.0%} of days.</p>"]
    REPORTS.mkdir(exist_ok=True)
    d.to_csv(REPORTS / f"{symbol}_backtest.csv", index=False)
    path = REPORTS / f"{symbol}_backtest.html"
    path.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                    f"<title>{html.escape(symbol)} backtest</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>", encoding="utf-8")
    return path
