"""Benchmarks: ordinary least squares with no regimes and equal weights."""
import numpy as np

from .base import Forecast, Model, design, winsorize


def _ols(y, r):
    X = design(r)
    yw = winsorize(y)
    theta = np.linalg.lstsq(X, yw, rcond=None)[0]
    return Forecast(*theta, resid_sd=float(np.std(yw - X @ theta)))


class RollingOlsModel(Model):
    """Regression on the last `window` observations; everything older is forgotten at once."""

    def __init__(self, window=60):
        self.window = window
        self.name = f"ols{window}"
        self.label = f"Rolling {window} observations"

    def forecast(self, y, r):
        return _ols(np.asarray(y, float)[-self.window:], np.asarray(r, float)[-self.window:])


class ExpandingOlsModel(Model):
    """Regression on all history: what you get if the relationship never changed."""
    name = "ols_all"
    label = "All history, one regime"

    def forecast(self, y, r):
        return _ols(np.asarray(y, float), np.asarray(r, float))
