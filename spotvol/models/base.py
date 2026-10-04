"""What every model shares: the regression design, the outlier cap, and the interface."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WINSOR = 4.0


def design(r, closed=None):
    """Columns: intercept, up part of the return, down part of the return (in %, down part negative), and, if given,
    the number of closed-market days the observation spans (2 for a normal weekend)."""
    r = np.asarray(r, float)
    cols = [np.ones_like(r), np.clip(r, 0, None), np.clip(r, None, 0)]
    if closed is not None:
        cols.append(np.asarray(closed, float))
    return np.column_stack(cols)


def winsorize(y, k=WINSOR):
    """Cap vol changes at k robust standard deviations so a single wild day cannot set the betas."""
    med = np.median(y)
    mad = 1.4826 * np.median(np.abs(y - med))
    return np.clip(y, med - k * mad, med + k * mad)


@dataclass
class Forecast:
    """A model's view of the next observation. Slopes are vol points per 1% of spot return:
    b_up > 0 means vol rises on a rally, b_dn < 0 means vol rises on a sell-off.
    c_closed is the vol change per closed-market day spanned (the calendar-clock effect); 0 for models without it."""
    a: float
    b_up: float
    b_dn: float
    resid_sd: float
    c_closed: float = 0.0
    detail: dict = field(default_factory=dict)     # model-specific extras (regime table, half-life chosen, ...)

    def predict(self, r, closed=0):
        r = np.asarray(r, float)
        return self.a + self.b_up * np.clip(r, 0, None) + self.b_dn * np.clip(r, None, 0) + self.c_closed * closed


def forecast_from(coef, resid_sd, detail=None):
    """Forecast from a coefficient row: a, b_up, b_dn and optionally c_closed."""
    return Forecast(coef[0], coef[1], coef[2], resid_sd=float(resid_sd), c_closed=float(coef[3]) if len(coef) > 3 else 0.0,
                    detail=detail or {})


class Model:
    name = ""       # short id, used in the database and on the command line
    label = ""      # shown in reports

    def forecast(self, y, r, closed=None, atm=None) -> Forecast:
        """Fit on the history given and forecast the next observation.
        y: vol changes; r: spot returns in %; closed: closed-market days spanned by each observation;
        atm: ATM vol in vol points at the start of each observation, with one extra entry for today's level
        (len(y) + 1). Models that do not use `closed` or `atm` ignore them."""
        raise NotImplementedError

    def walk_forward(self, y, r, start, step=1, closed=None, atm=None) -> pd.DataFrame:
        """Out-of-sample path. Row t holds the coefficients fitted on observations before t and the prediction
        for t given its spot return r[t] (and closed[t]). Rows before `start` are NaN. Coefficients are refitted
        every `step` observations. Models with a faster way to do this override it."""
        y, r = np.asarray(y, float), np.asarray(r, float)
        out = np.full((len(y), 4), np.nan)
        for t0 in range(start, len(y), step):
            f = self.forecast(y[:t0], r[:t0], None if closed is None else closed[:t0], None if atm is None else atm[:t0 + 1])
            out[t0:t0 + step] = (f.a, f.b_up, f.b_dn, f.c_closed)
        return path_frame(out, r, closed)


def path_frame(coef, r, closed=None):
    """coef: array with columns a, b_up, b_dn and optionally c_closed per observation -> frame with the prediction."""
    coef = np.asarray(coef, float)
    if coef.shape[1] == 3:
        coef = np.column_stack([coef, np.zeros(len(coef))])
    df = pd.DataFrame(coef, columns=["a", "b_up", "b_dn", "c_closed"])
    c = np.zeros(len(r)) if closed is None else np.asarray(closed, float)
    df["pred"] = (coef * design(r, c)).sum(axis=1)
    return df
