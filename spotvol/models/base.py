"""What every model shares: the regression design, the outlier cap, and the interface."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WINSOR = 4.0


def design(r):
    """Columns: intercept, up part of the return, down part of the return (in %, down part negative)."""
    r = np.asarray(r, float)
    return np.column_stack([np.ones_like(r), np.clip(r, 0, None), np.clip(r, None, 0)])


def winsorize(y, k=WINSOR):
    """Cap vol changes at k robust standard deviations so a single wild day cannot set the betas."""
    med = np.median(y)
    mad = 1.4826 * np.median(np.abs(y - med))
    return np.clip(y, med - k * mad, med + k * mad)


@dataclass
class Forecast:
    """A model's view of the next observation. Slopes are vol points per 1% of spot return:
    b_up > 0 means vol rises on a rally, b_dn < 0 means vol rises on a sell-off."""
    a: float
    b_up: float
    b_dn: float
    resid_sd: float
    detail: dict = field(default_factory=dict)     # model-specific extras (regime table, half-life chosen, ...)

    def predict(self, r):
        r = np.asarray(r, float)
        return self.a + self.b_up * np.clip(r, 0, None) + self.b_dn * np.clip(r, None, 0)


class Model:
    name = ""       # short id, used in the database and on the command line
    label = ""      # shown in reports

    def forecast(self, y, r) -> Forecast:
        """Fit on the history given (y = vol changes, r = spot returns) and forecast the next observation."""
        raise NotImplementedError

    def walk_forward(self, y, r, start, step=1) -> pd.DataFrame:
        """Out-of-sample path. Row t holds the coefficients fitted on observations before t and the prediction
        for t given its spot return r[t]. Rows before `start` are NaN. Coefficients are refitted every `step`
        observations. Models with a faster way to do this override it."""
        y, r = np.asarray(y, float), np.asarray(r, float)
        out = np.full((len(y), 3), np.nan)
        for t0 in range(start, len(y), step):
            f = self.forecast(y[:t0], r[:t0])
            out[t0:t0 + step] = (f.a, f.b_up, f.b_dn)
        return path_frame(out, r)


def path_frame(coef, r):
    """coef: array with columns a, b_up, b_dn per observation -> frame with the prediction added."""
    df = pd.DataFrame(coef, columns=["a", "b_up", "b_dn"])
    df["pred"] = (coef * design(r)).sum(axis=1)
    return df
