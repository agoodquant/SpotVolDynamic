"""HTML reports: one per name, plus a cross-sectional screen."""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import analysis as an
from . import monitor as mon
from . import store
from .models import MODELS, PRIMARY, get
from .config import REPORTS, TENORS

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
SURFACE, GRID, AXIS, INK, INK2 = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

CSS = f"""
body {{ font-family: {FONT}; background: #f9f9f7; color: {INK}; margin: 0; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 36px 0 6px; }}
p, li {{ color: {INK2}; line-height: 1.5; max-width: 860px; }}
.verdict {{ color: {INK}; font-size: 17px; }}
.tiles {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
.tile {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,.1); border-radius: 8px; padding: 12px 16px; min-width: 150px; }}
.tile b {{ display: block; font-size: 24px; font-weight: 600; }} .tile span {{ color: {INK2}; font-size: 13px; }}
.card {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,.1); border-radius: 8px; padding: 8px; overflow-x: auto; }}
table {{ border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; width: 100%; }}
th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid {GRID}; white-space: nowrap; }}
th {{ color: {INK2}; font-weight: 600; }} th:first-child, td:first-child {{ text-align: left; }}
"""

COLS = {"sample": "Sample", "n": "Obs", "beta_atm": "β ATM", "t_atm": "t", "beta_fix": "β fixed strike", "t_fix": "t ",
        "corr_fix": "Corr", "skew_slope": "Skew slope", "beta_fix_up": "β up days", "beta_fix_down": "β down days",
        "horizon_obs": "Horizon (obs)", "bucket": "Spot move", "avg_ret": "Avg move %", "avg_dAtm": "Avg ΔATM vol",
        "avg_dFix": "Avg Δfixed-strike vol", "vol_up_freq": "Vol up", "tenor": "Tenor", "symbol": "Name",
        "atm": "ATM vol", "rr25": "25Δ RR (put−call)", "date": "As of", "regime": "Regime", "b_up": "β up", "t_up": "t", "b_dn": "β down", "t_dn": "t ", "share": "Share of history", "stay_prob": "Prob. of staying", "duration": "Avg length (obs)", "prob_now": "Prob. today", "prob_next": "Prob. next day", "prob": "Prob.", "move": "Next-day spot move", "exp_dvol": "Expected Δvol, vol pts", "sd_dvol": "± 1 s.d.", "model": "Model", "oos_r2": "R², all days", "oos_r2_up": "R², up days", "oos_r2_down": "R², down days", "b_up_next": "β up, next day", "b_dn_next": "β down, next day", "resid_sd": "One-day noise, vol pts", "c_closed": "Per closed day", "crash_coef": "Crash β", "crash_threshold": "Crash beyond, %", "a_next": "Drift per trading day", "beta_fix_3m": "β fixed strike, 3m", "t_fix_3m": "t, 3m"}
FMT = {"n": "{:.0f}", "resid_sd": "{:.2f}", "c_closed": "{:+.2f}", "crash_coef": "{:+.2f}", "crash_threshold": "{:.1f}", "a_next": "{:+.2f}", "t_up": "{:.1f}", "t_dn": "{:.1f}", "share": "{:.0%}", "stay_prob": "{:.0%}", "duration": "{:.0f}", "prob_now": "{:.0%}", "prob_next": "{:.0%}", "prob": "{:.0%}", "exp_dvol": "{:+.2f}", "sd_dvol": "{:.2f}", "oos_r2": "{:+.1%}", "oos_r2_up": "{:+.1%}", "oos_r2_down": "{:+.1%}", "t_fix_3m": "{:.1f}", "horizon_obs": "{:.0f}", "t_atm": "{:.1f}", "t_fix": "{:.1f}", "corr_fix": "{:.2f}",
       "vol_up_freq": "{:.0%}", "avg_ret": "{:.1f}", "avg_dAtm": "{:+.2f}", "avg_dFix": "{:+.2f}", "atm": "{:.1f}", "rr25": "{:+.1f}"}


def _table(df):
    head = "".join(f"<th>{html.escape(COLS.get(c, c))}</th>" for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        cells = ""
        for c in df.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                v = "–" if np.isnan(v) else FMT.get(c, "{:+.3f}").format(v)
            elif isinstance(v, pd.Timestamp):
                v = f"{v:%Y-%m-%d}"
            cells += f"<td>{html.escape(str(v))}</td>"
        body += f"<tr>{cells}</tr>"
    return f'<div class="card"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _style(fig, height):
    fig.update_layout(height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, color=INK2, size=13),
                      margin=dict(l=60, r=24, t=36, b=40), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0, font=dict(color=INK)))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS)
    return fig


def _fig_levels(lv):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Forward, split-adjusted (log scale)", "ATM implied vol, %", "25-delta risk reversal, put minus call, vol pts"))
    fig.add_scatter(x=lv["date"], y=lv["F_adj"], line=dict(color=BLUE, width=2), name="Forward", row=1, col=1)
    fig.add_scatter(x=lv["date"], y=100 * lv["atm"], line=dict(color=BLUE, width=2), name="ATM vol", row=2, col=1)
    fig.add_scatter(x=lv["date"], y=100 * lv["rr25"], line=dict(color=BLUE, width=2), name="25Δ RR", row=3, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_layout(showlegend=False)
    return _style(fig, 620)


def _fig_rolling(ro):
    fig = go.Figure()
    fig.add_scatter(x=ro["date"], y=ro["beta_fix"], line=dict(color=BLUE, width=2), name="Realized β, fixed strike")
    fig.add_scatter(x=ro["date"], y=ro["beta_atm"], line=dict(color=ORANGE, width=2), name="Realized β, ATM")
    fig.add_scatter(x=ro["date"], y=ro["skew_slope"], line=dict(color=AQUA, width=2), name="Skew slope (sticky-strike prediction for ATM)")
    fig.update_yaxes(title="vol pts per 1% spot move")
    return _style(fig, 380)


def _fig_scatter(p, cut):
    fig = go.Figure()
    for label, g, color in ((f"Before {cut:%b %Y}", p[p["d1"] <= cut], MUTED), (f"Since {cut:%b %Y}", p[p["d1"] > cut], BLUE)):
        if len(g) < 8:
            continue
        fig.add_scatter(x=g["r"], y=g["dFix"], mode="markers", name=label, text=g["d1"].dt.strftime("%Y-%m-%d"),
                        marker=dict(color=color, size=8, opacity=0.75, line=dict(color=SURFACE, width=1)),
                        hovertemplate="%{text}<br>spot %{x:.1f}%<br>vol %{y:+.2f} pts<extra></extra>")
        o = an.ols(g["r"], g["dFix"])
        xs = np.array([g["r"].min(), g["r"].max()])
        fig.add_scatter(x=xs, y=g["dFix"].mean() + o["beta"] * (xs - g["r"].mean()), mode="lines",
                        line=dict(color=color, width=2), showlegend=False, hoverinfo="skip")
    fig.update_xaxes(title="Spot move, %")
    fig.update_yaxes(title="Change in fixed-strike vol, vol pts")
    return _style(fig, 420).update_layout(hovermode="closest")


def _fig_buckets(b):
    b = b[b["n"] >= 5]      # the table below keeps the thin buckets
    fig = go.Figure(go.Bar(x=b["bucket"].astype(str), y=b["avg_dFix"], width=0.45, marker=dict(color=BLUE, cornerradius=4),
                           customdata=np.c_[b["n"], b["vol_up_freq"]],
                           hovertemplate="%{x}<br>avg %{y:+.2f} vol pts<br>%{customdata[0]:.0f} obs, vol up %{customdata[1]:.0%}<extra></extra>"))
    fig.update_xaxes(title="Spot move")
    fig.update_yaxes(title="Average change in fixed-strike vol, vol pts")
    return _style(fig, 360).update_layout(hovermode="closest")


def _fig_regime(dates, smooth, name):
    # two regimes are mirror images, so one line carries everything
    fig = go.Figure(go.Scatter(x=dates, y=smooth[:, -1], line=dict(color=BLUE, width=2), name=name, fill="tozeroy", fillcolor="rgba(42,120,214,0.10)"))
    fig.update_yaxes(title="Probability", range=[0, 1], tickformat=".0%")
    return _style(fig, 300).update_layout(showlegend=False, title=dict(text=f"Probability of regime {name}", font=dict(size=14, color=INK), x=0.01))


def _monitor_section(symbol, ten):
    m = mon.monitor(symbol, ten)
    if m is None:
        return []
    f = m["primary"]
    scen = m["scenarios"].copy()
    for c in scen.columns[1:]:
        scen[c] = scen[c].map("{:+.2f}".format)
    out = ["<h2>Next-day asymmetric β</h2>",
           f"<p>The {html.escape(get(PRIMARY).label.lower())} model implies β up = {f.b_up:+.2f} and β down = {f.b_dn:+.2f} for the next observation. "
           "Both are slopes of fixed-strike vol on the spot return, in vol points per 1%: a positive β up means vol rises on a rally, "
           "a negative β down means vol rises on a sell-off. The models differ in how fast they forget old behaviour. "
           "Vendor vols run on a calendar-day clock, so they drift down on ordinary days and jump after a weekend: "
           f"the per-closed-day column is that jump (the primary model's is {f.c_closed:+.2f} per day, so {2 * f.c_closed:+.2f} over a normal weekend), "
           "and the drift column is the rest. The Markov model has no weekend term. The crash model splits a sell-off at 2 daily standard deviations: "
           "β down applies to the first part, the crash β (estimated with a one-year memory) to the rest; other models leave those columns blank.</p>", _table(m["table"]),
           f"<p>Expected change in fixed-strike vol for a given next-day spot move, by model, in vol points. The next observation spans "
           f"{m['next_closed']} closed day{'' if m['next_closed'] == 1 else 's'}{' (a weekend)' if m['next_closed'] == 2 else ''}, and the figures include that. "
           "For the spot/vol charge itself use the β; multiply by vega to get the vol P&L. "
           "The one-day noise in the table above is the uncertainty around it.</p>", _table(scen)]
    d = m["forecasts"]["markov"].detail
    bic = d["bic"]
    if d["n_regimes"] == 1:
        out += [f"<p>Regime-switching model: two regimes do not fit better than one (BIC {bic[1]:.0f} for one, {bic[2]:.0f} for two).</p>"]
    else:
        out += [f"<p>Regime-switching model: two regimes fit better than one (BIC {bic[2]:.0f} against {bic[1]:.0f}; a gap under about 6 is weak evidence). "
                f"Today it puts {d['prob']:.0%} on <b>{html.escape(d['regime'])}</b>.</p>", _table(d["regimes"]),
                f'<div class="card">{_fig_regime(m["dates"], d["smooth"], d["regimes"]["regime"].iloc[-1]).to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</div>']
    bt = REPORTS / f"{symbol}_backtest.html"
    out += [f'<p>How accurate these models have been out of sample: <a href="{bt.name}">backtest report</a>.</p>' if bt.exists()
            else f"<p>To see how accurate these models have been out of sample, run <code>python -m spotvol backtest {symbol}</code>.</p>"]
    return out


def _primary(res):
    return max(res["tenors"], key=lambda k: len(res["tenors"][k]["pairs"]))


def _last(p, days):
    return p[p["d1"] > p["d1"].max() - pd.Timedelta(days=days)]


def _verdict(s):
    if np.isnan(s["t_fix"]):
        return "Not enough observations in the last 6 months to judge."
    if s["t_fix"] > 2:
        head = "Vol rises with spot"
    elif s["t_fix"] < -2:
        head = "Vol falls as spot rises (the normal equity pattern)"
    else:
        head = "No significant spot/vol link at fixed strike"
    return (f"<b>{head}</b> over the last 6 months: fixed-strike vol moved {s['beta_fix']:+.2f} vol pts per 1% spot move "
            f"(t = {s['t_fix']:.1f}); ATM vol moved {s['beta_atm']:+.2f}, against {s['skew_slope']:+.2f} implied by the skew under sticky strike.")


def _tenor_table(res):
    rows = []
    for name, t in res["tenors"].items():
        for label, p in (("last 6 months", _last(t["pairs"], 182)), ("full sample", t["pairs"])):
            if len(p) >= 20:
                rows.append({"tenor": f"{name}, {label}", **an.summary(p)})
    return pd.DataFrame(rows)


def write_report(symbol, bench=None):
    res = an.analyse(symbol)
    if res is None or not res["tenors"]:
        raise SystemExit(f"no data for {symbol}: run backfill/snapshot and build first")
    ten = _primary(res)
    t = res["tenors"][ten]
    p, lv = t["pairs"], t["levels"]
    s6 = an.summary(_last(p, 182))
    last = lv.iloc[-1]
    figs = [_fig_levels(lv), _fig_rolling(an.rolling(p)), _fig_scatter(p, p["d1"].max() - pd.Timedelta(days=182)), _fig_buckets(an.buckets(_last(p, 365)))]
    divs = [f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False, config={"displaylogo": False}) for i, f in enumerate(figs)]

    tiles = [(f"{100 * last['atm']:.1f}", f"ATM vol, {ten}"), (f"{100 * last['rr25']:+.1f}", "25Δ RR, put−call, vol pts"),
             (f"{s6['skew_slope']:+.2f}", "Skew slope, 6m avg"), (f"{s6['beta_fix']:+.2f}", f"β fixed strike, 6m (t = {s6['t_fix']:.1f})"),
             (f"{s6['beta_atm']:+.2f}", "β ATM, 6m")]
    sm = res["smiles"]
    src = ", ".join(f"{k}: {g['date'].nunique()} dates ({g['date'].min():%Y-%m-%d} to {g['date'].max():%Y-%m-%d}), expiries to {g['dte'].max()} days"
                    for k, g in sm.groupby("source"))
    earn_note = (f"{t['n_earn']} observations spanning earnings were excluded (Yahoo earnings calendar)." if len(res["earnings"])
                 else "No earnings calendar found for this name (normal for ETFs), so nothing was excluded.")

    parts = [f"<h1>{symbol} spot/vol dynamics</h1><p>As of {last['date']:%Y-%m-%d}. Primary tenor: {ten} (expiry nearest {TENORS[ten]} days).</p>",
             f'<p class="verdict">{_verdict(s6)}</p>',
             '<div class="tiles">' + "".join(f'<div class="tile"><b>{v}</b><span>{html.escape(k)}</span></div>' for v, k in tiles) + "</div>",
             "<p>β is the change in implied vol, in vol points, per 1% spot move for the same expiry. Fixed-strike β is zero if the smile simply "
             "stays put (sticky strike); ATM β then equals the skew slope. A fixed-strike β well above zero is vol moving with spot beyond anything the skew prices.</p>",
             "<h2>Spot, vol and skew</h2>", f'<div class="card">{divs[0]}</div>',
             "<h2>Rolling 60-observation β against the skew</h2>", f'<div class="card">{divs[1]}</div>',
             "<h2>Daily vol change against spot move</h2>", f'<div class="card">{divs[2]}</div>',
             "<h2>Average vol change by size of spot move, last 12 months</h2>", f'<div class="card">{divs[3]}</div>', _table(an.buckets(_last(p, 365))),
             *_monitor_section(symbol, ten),
             "<h2>By calendar period</h2>", _table(an.regimes(p)),
             "<h2>By horizon, last 12 months</h2><p>Non-overlapping windows of consecutive observations.</p>", _table(an.horizons(_last(p, 365))),
             "<h2>By tenor</h2><p>Longer tenors need the daily Yahoo snapshots to accumulate; they appear here once there are 20 observations.</p>", _table(_tenor_table(res))]
    if bench:
        rb = an.analyse(bench)
        if rb and ten in rb["tenors"]:
            pb = rb["tenors"][ten]["pairs"]
            cmp_ = pd.DataFrame([{"sample": f"{n_}, {lab}", **an.summary(_last(q, d))}
                                 for lab, d in (("last 6 months", 182), ("last 12 months", 365)) for n_, q in ((symbol, p), (bench, pb))])
            parts += [f"<h2>Against {bench}</h2>", _table(cmp_)]
    parts += ["<h2>Data</h2>", f"<p>{html.escape(src)}.<br>{earn_note}<br>Implied vols are computed from bid/ask mids on out-of-the-money options "
              "against the put-call parity forward. Both sources are free and unofficial; treat single-day outliers with suspicion.</p>"]

    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{symbol}.html"
    path.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                    f"<title>{symbol} spot/vol</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>", encoding="utf-8")
    return path


def write_screen(symbols, tenor="1m"):
    rows = []
    for s in symbols:
        res = an.analyse(s)
        if res is None or tenor not in res["tenors"]:
            continue
        t = res["tenors"][tenor]
        last = t["levels"].iloc[-1]
        s6, s3 = an.summary(_last(t["pairs"], 182)), an.summary(_last(t["pairs"], 91))
        rows.append({"symbol": s, "date": last["date"], "atm": 100 * last["atm"], "rr25": 100 * last["rr25"], "skew_slope": s6["skew_slope"],
                     "beta_atm": s6["beta_atm"], "beta_fix": s6["beta_fix"], "t_fix": s6["t_fix"],
                     "n": s6["n"]})
        m = mon.monitor(s, tenor)
        if m:
            store.write_monitor(mon.rows_for_store(m))
            d = m["forecasts"]["markov"].detail
            rows[-1].update({"b_up_next": m["primary"].b_up, "b_dn_next": m["primary"].b_dn, "regime": d["regime"], "prob": d["prob"]})
    df = pd.DataFrame(rows).sort_values("beta_fix", ascending=False)
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "screen.csv", index=False)
    body = (f"<h1>Spot/vol screen</h1><p>Tenor {tenor}, last 6 months unless stated, earnings excluded. Sorted by fixed-strike β: names at the top are those "
            "where vol rises with spot beyond what the skew prices. β in vol points per 1% spot move. β up and β down for the next day come from the short-memory model; the regime is the regime-switching model's current call.</p>" + _table(df))
    path = REPORTS / "screen.html"
    path.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                    f"<title>Spot/vol screen</title><style>{CSS}</style></head><body><main>{body}</main></body></html>", encoding="utf-8")
    return path


def write_index():
    """reports/index.html: one row per name with links to its report and backtest, and the screen's headline numbers.
    Rebuilt from whatever report files exist, so it is safe to call after any write."""
    screen = pd.read_csv(REPORTS / "screen.csv", parse_dates=["date"]).set_index("symbol") if (REPORTS / "screen.csv").exists() else pd.DataFrame()
    names = sorted(p.stem for p in REPORTS.glob("*.html") if p.stem not in ("index", "screen") and not p.stem.endswith("_backtest"))
    rows = ""
    for s in names:
        bt = REPORTS / f"{s}_backtest.html"
        cells = [f'<a href="{s}.html">{s}</a>', f'<a href="{bt.name}">Backtest</a>' if bt.exists() else "–"]
        if s in screen.index:
            r = screen.loc[s]
            cells += [f"{r['date']:%Y-%m-%d}", f"{r['atm']:.1f}", f"{r['beta_fix']:+.2f}", f"{r['b_up_next']:+.2f}", f"{r['b_dn_next']:+.2f}",
                      html.escape(str(r["regime"]))]
        else:
            cells += ["–"] * 6
        rows += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    head = "".join(f"<th>{h}</th>" for h in ("Name", "Backtest", "As of", "ATM vol", "β fixed strike, 6m", "β up, next day", "β down, next day", "Regime (Markov)"))
    body = (f"<h1>Spot/vol reports</h1><p>Start with the <a href=\"screen.html\">screen</a>, which ranks every name by how much its vol moves with spot "
            "beyond what the skew prices. β in vol points per 1% spot move; next-day betas from the short-memory model.</p>"
            f'<div class="card"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
            f"<p>Index rebuilt {pd.Timestamp.now():%Y-%m-%d %H:%M}. Names without screen numbers are not in <code>universe.yaml</code>.</p>")
    path = REPORTS / "index.html"
    path.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                    f"<title>Spot/vol reports</title><style>{CSS}a {{ color: {BLUE}; }} th:nth-child(2), td:nth-child(2), th:last-child, td:last-child {{ text-align: left; }}</style></head><body><main>{body}</main></body></html>", encoding="utf-8")
    return path
