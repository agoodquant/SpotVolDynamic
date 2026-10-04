"""Short-memory betas plus a long-memory crash term.

    dvol = a + b_up * max(r, 0) + b_dn * max(min(r, 0), -k*s) + g * min(r + k*s, 0) + c_closed * closed_days + noise

s is the stock's daily return standard deviation, estimated from past returns only (exponentially weighted,
half-life `sd_halflife`). A sell-off is split at k standard deviations: b_dn prices the first k*s of the drop, g the part
beyond it. With k = 2 that is about -6% for MU and -3% for JPM, so "crash" means the same thing for every name.

Crash days are rare (MU: 36 drops of 5% or more in nearly three years, in about 16 episodes), so a 20-observation memory
sees only two or three observations' worth of them. The coefficients are therefore estimated in two stages, each day,
from earlier data only:

  1. long memory (`crash_halflife`, a year by default): fit every coefficient, keep only g;
  2. short memory (`halflife`): remove g * crash part from each vol change, fit a, b_up, b_dn and c_closed.

g < 0 means vol rises on a crash, the same sign convention as b_dn.
"""
import numpy as np
import pandas as pd

from .base import Forecast, Model
from .ewm import ewm_path


def causal_sd(r, halflife, warmup=20):
    """Daily return standard deviation (in %) known before each observation; index len(r) is the value for the next one."""
    r = np.asarray(r, float)
    lam = 0.5 ** (1 / halflife)
    out = np.full(len(r) + 1, np.nan)
    v = np.mean(r[:warmup] ** 2) if len(r) else np.nan
    for t in range(len(r) + 1):
        out[t] = np.sqrt(v)
        if warmup <= t < len(r):
            v = lam * v + (1 - lam) * r[t] ** 2
    out[:warmup] = np.sqrt(np.mean(r[:warmup] ** 2))       # the warm-up period borrows its own average; not used for prediction
    return out


def split_return(r, thresh):
    """Return the regression columns up, ordinary down (floored at -thresh) and crash (beyond -thresh, <= 0)."""
    r, thresh = np.asarray(r, float), np.asarray(thresh, float)
    return np.clip(r, 0, None), np.maximum(np.clip(r, None, 0), -thresh), np.clip(r + thresh, None, 0)


class CrashEwmModel(Model):
    def __init__(self, halflife=20, crash_halflife=250, k=2.0, sd_halflife=60, prior_obs=5):
        self.halflife, self.crash_halflife, self.k, self.sd_halflife, self.prior_obs = halflife, crash_halflife, k, sd_halflife, prior_obs
        self.name = f"ewm{halflife}_crash"
        self.label = f"Short memory, half-life {halflife}, with crash term"

    def _fit(self, y, r, closed):
        y, r = np.asarray(y, float), np.asarray(r, float)
        closed = np.zeros(len(r)) if closed is None else np.asarray(closed, float)
        thresh = self.k * causal_sd(r, self.sd_halflife)                     # len(r) + 1
        up, dn, cr = split_return(r, thresh[:-1])
        one = np.ones(len(r))
        long_coef, _ = ewm_path(y, np.column_stack([one, up, dn, cr, closed]), self.crash_halflife, self.prior_obs)
        g = np.nan_to_num(long_coef[:, 3])                                    # crash coefficient known before each obs
        short_coef, sd = ewm_path(y - g[:-1] * cr, np.column_stack([one, up, dn, closed]), self.halflife, self.prior_obs)
        return short_coef, g, thresh, sd, (up, dn, cr, closed)

    def forecast(self, y, r, closed=None):
        coef, g, thresh, sd, _ = self._fit(y, r, closed)
        a, b_up, b_dn, c = coef[-1]
        return CrashForecast(a, b_up, b_dn, resid_sd=float(sd[-1]), c_closed=float(c), g=float(g[-1]), thresh=float(thresh[-1]),
                             detail={"halflife": self.halflife, "crash_coef": float(g[-1]), "crash_threshold": float(thresh[-1])})

    def walk_forward(self, y, r, start, step=1, closed=None):
        coef, g, thresh, _, (up, dn, cr, cl) = self._fit(y, r, closed)
        coef, g, thresh = coef[:-1].copy(), g[:-1].copy(), thresh[:-1].copy()
        coef[:start], g[:start], thresh[:start] = np.nan, np.nan, np.nan
        df = pd.DataFrame(coef, columns=["a", "b_up", "b_dn", "c_closed"])
        df["crash_coef"], df["crash_threshold"] = g, thresh
        df["pred"] = coef[:, 0] + coef[:, 1] * up + coef[:, 2] * dn + coef[:, 3] * cl + g * cr
        return df


class CrashForecast(Forecast):
    """A Forecast whose down side is split at -thresh: b_dn up to it, g beyond it."""

    def __init__(self, a, b_up, b_dn, resid_sd, c_closed=0.0, g=0.0, thresh=np.inf, detail=None):
        super().__init__(a, b_up, b_dn, resid_sd, c_closed, detail or {})
        self.g, self.thresh = g, thresh

    def predict(self, r, closed=0):
        up, dn, cr = split_return(r, self.thresh)
        return self.a + self.b_up * up + self.b_dn * dn + self.g * cr + self.c_closed * closed
