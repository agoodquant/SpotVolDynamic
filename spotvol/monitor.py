"""Today's view from every model: next-day asymmetric betas and the vol change they imply for a given spot move."""
import pandas as pd

from . import analysis as an
from .models import MODELS, PRIMARY

SCENARIOS = [-5, -3, -1, 1, 3, 5]
MIN_OBS = 120


def monitor(symbol, tenor="1m", target="dFix"):
    """target: 'dFix' (fixed-strike vol change) or 'dAtm' (ATM vol change)."""
    res = an.analyse(symbol)
    if res is None or tenor not in res["tenors"]:
        return None
    p = res["tenors"][tenor]["pairs"]
    if len(p) < MIN_OBS:
        return None
    y, r, closed = p[target].to_numpy(), p["r"].to_numpy(), p["closed"].to_numpy()
    nxt_closed = an.next_closed_days(p["d1"].iloc[-1])
    forecasts = {m.name: m.forecast(y, r, closed) for m in MODELS}
    table = pd.DataFrame([{"model": m.label, "b_up_next": f.b_up, "b_dn_next": f.b_dn, "c_closed": f.c_closed,
                           "a_next": f.a, "resid_sd": f.resid_sd} for m in MODELS for f in [forecasts[m.name]]])
    scenarios = pd.DataFrame({"move": [f"{s:+d}%" for s in SCENARIOS],
                              **{m.label: forecasts[m.name].predict(SCENARIOS, nxt_closed) for m in MODELS}})
    return {"symbol": symbol, "tenor": tenor, "target": target, "date": p["d1"].iloc[-1], "dates": p["d1"].to_numpy(), "n": len(p),
            "forecasts": forecasts, "primary": forecasts[PRIMARY], "table": table, "scenarios": scenarios, "next_closed": nxt_closed}


def rows_for_store(m):
    """One row per model for the beta_monitor table."""
    out = []
    for name, f in m["forecasts"].items():
        out.append({"date": m["date"], "symbol": m["symbol"], "tenor": m["tenor"], "target": m["target"], "model": name,
                    "a_next": f.a, "b_up_next": f.b_up, "b_dn_next": f.b_dn, "c_closed": f.c_closed, "resid_sd": f.resid_sd,
                    "n_regimes": f.detail.get("n_regimes"), "regime": f.detail.get("regime"), "prob": f.detail.get("prob"),
                    "halflife": f.detail.get("halflife")})
    return out
