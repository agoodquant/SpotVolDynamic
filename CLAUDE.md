# SpotVolDynamic

Measures how implied vol moves when spot moves, name by name, and compares it with what the skew prices.
Built for an equity exotics desk (autocalls on single stocks): the concern is names such as MU where the skew
is flat but vol rises on rallies, which a local vol model calibrated to that skew prices at zero.

## Running things

There is no Python on PATH. Always use the project virtualenv, from the project root:

```
.venv\Scripts\python.exe -m spotvol <command> [SYMBOLS] [--bench SPY]
```

| Command | What it does |
|---|---|
| `snapshot` | Store today's full Yahoo option chain (all expiries) |
| `backfill` | Pull DoltHub history, new dates only |
| `build` | Raw quotes to smiles, new dates only (`--rebuild` to redo) |
| `report SYM --bench SPY` | Write `reports/SYM.html`; fetches data for a new name automatically |
| `screen` | Rank the universe, write `reports/screen.html` and store each model's forecast |
| `backtest SYM` | Rolling out-of-sample test of every model, about 2 minutes per name |
| `daily` | snapshot + backfill + build + screen for `universe.yaml` |

With no symbols, commands run over `universe.yaml`. A Windows scheduled task ("SpotVol daily snapshot",
`scripts/register_task.ps1`) runs `daily` at 15:45 New York time on weekdays.

There are no unit tests. Verify a change by running `report` or `backtest` on MU and SPY and reading the numbers:
SPY should show ATM beta near -1 and a beta-to-skew ratio near 1.5; MU should show a flat skew.

## Layout

```
spotvol/
  config.py     paths, tenors, source preference
  store.py      DuckDB access; the only module that touches the database
  dolthub.py    history source        yahoo.py   snapshot source and earnings calendar
  smile.py      forward, implied vols, ATM vol, 25-delta skew per (date, expiry)
  analysis.py   pairs of consecutive days, regressions by regime, horizon, bucket
  models/       one file per beta model, common interface in base.py (see models/__init__.py)
  monitor.py    today's forecast from every model
  backtest.py   rolling out-of-sample test over every model
  report.py     HTML reports and the screen
  cli.py        command line
```

To add a model: one new file in `spotvol/models/` subclassing `base.Model` with a `forecast(y, r)` method,
then add it to `MODELS` in `models/__init__.py`. Monitor, screen and backtest pick it up from that list.

## Data

Everything is in one DuckDB file, `data/spotvol.duckdb` (not in git). Tables: `raw_quotes`, `smiles`,
`earnings`, `beta_monitor`, `empty_dates`.

- **DoltHub** (`post-no-preference/options`): end-of-day history, but only three front expiries (to about
  7 weeks). Query it by exact dates (`date in (...)`), one week at a time. Range queries and
  `count(*)` without a date time out. The API is flaky; failed dates are retried on the next run.
  It has SPY but not SPX, NDX, QQQ, IWM or VIX.
- **Yahoo**: today's chain only, no history, so a missed day is lost. All expiries including LEAPS.
  Index symbols are `^SPX`, `^NDX`. Yahoo's own implied vols are not used.
- Raw quotes are immutable: a (source, symbol, date) slice is written once. Smiles are derived and can be rebuilt.
- DuckDB allows one writing process. Do not run two commands that write at the same time.

## Conventions that are easy to get wrong

- **A "pair"** is two consecutive observation dates and the same listed expiry on both. Returns use that
  expiry's put-call parity forward.
- **beta** is vol points per 1% spot move. `dAtm` is the ATM vol change; `dFix` is the change in vol at the
  first day's ATM strike. `beta_atm ~ beta_fix + skew_slope`.
- **Skew slope** is d(vol)/d(ln K) between the 25-delta put and call, so it is negative for a normal put skew.
  `rr25` is put minus call, so it is positive for a normal put skew.
- **Asymmetric betas**: `b_up` and `b_dn` are both slopes on the signed return. `b_up > 0` means vol rises on a
  rally; `b_dn < 0` means vol rises on a sell-off.
- **Excluded pairs**: those spanning earnings (Yahoo calendar; after-close versus before-open matters),
  moves above 35% (treated as splits), and stale vendor duplicates.
- **Tenor buckets** accept expiries within 0.65x to 1.6x of the target days. Only the 1m bucket has real
  history; longer tenors fill in as Yahoo snapshots accumulate.
- Model fitting caps vol changes at 4 robust standard deviations. Backtest scoring uses the uncapped values.
- Backtests must stay causal: a prediction for day t may use the spot move on day t but nothing else from
  day t onward. `ewm.causal_winsorize` exists for this reason.

## Known findings, so they are not re-derived

- Vendor vols use a calendar-day clock, so fixed-strike vol drifts down on ordinary days and jumps after weekends
  (MU: about +1.5 vol pts on Mondays). Every model except Markov has a `closed` term (closed-market days spanned,
  2 for a weekend) to absorb this; without it the intercept carried the effect as a large negative drift.
  `analysis.closed_days` ignores weekday holidays.
- Out-of-sample accuracy is still low: MU about 11% R-squared with the closed term (6% without), mean absolute
  error 1.85 against 1.92 for predicting no change. The up-day effect is real; down-day predictions are weaker.
- The short-memory model with half-life 20 is the primary model. That half-life was chosen on the same data
  it is tested on; 10 to 80 give similar results, 5 is too short.
- Tried and not better than the short-memory model: Markov regimes (recognises changes late; real-time call
  matches hindsight on about 57% of MU days), a Kalman filter with drifting betas, an adaptive half-life,
  vol mean reversion, term-structure roll-down, yesterday's vol change, and moves scaled by vol level.
- One beta for both directions (`EwmModel(symmetric=True)`): its apparent R-squared win on MU came from one day
  (9 Apr 2025, +17% rally, vol -15). Without it the asymmetric model is better; on down days one beta is worst.
- Crash term (`CrashEwmModel`, drops beyond k daily s.d., one-year memory): right sign for 9 of 10 names but no
  overall gain; only 2-11 crash days per name in the backtest. k=1.5 beats k=2 on MU. The threshold uses realized
  s.d., which put MU's split at about -9.5%; scaling by implied vol (about -6%) is untested.
- Backtest R-squared is fragile to single extreme days; the accuracy table also shows it without the 3 largest moves.
- Untested and most promising next: pooling betas, or crash days, across related names (semis).

## Reports and charts

Reports are self-contained HTML in `reports/`. Charts use Plotly with a fixed palette defined at
the top of `report.py`; no dual-axis charts, and at most three series per chart. When printing tables to the
Windows console, set `PYTHONIOENCODING=utf-8` (column names contain Greek letters).
