"""Benchmarks: ordinary least squares with no regimes and equal weights. Both include the closed-day term."""
import numpy as np

from .base import Model, design, forecast_from, winsorize


def _ols(y, r, closed=None):
    X = design(r, closed)
    yw = winsorize(y)
    theta = np.linalg.lstsq(X, yw, rcond=None)[0]
    return forecast_from(theta, np.std(yw - X @ theta))


class RollingOlsModel(Model):
    """Regression on the last `window` observations; everything older is forgotten at once."""

    def __init__(self, window=60):
        self.window = window
        self.name = f"ols{window}"
        self.label = f"Rolling {window} observations"

    def forecast(self, y, r, closed=None):
        w = self.window
        return _ols(np.asarray(y, float)[-w:], np.asarray(r, float)[-w:], None if closed is None else np.asarray(closed, float)[-w:])


class ExpandingOlsModel(Model):
    """Regression on all history: what you get if the relationship never changed."""
    name = "ols_all"
    label = "All history, one regime"

    def forecast(self, y, r, closed=None):
        return _ols(np.asarray(y, float), np.asarray(r, float), closed)
