"""Short-memory model: exponentially weighted regression.

One set of coefficients, re-estimated every day, with an observation's weight halving every `halflife` observations:

    weight of an observation k days old = 0.5 ** (k / halflife)

    dvol = a + b_up * max(r, 0) + b_dn * min(r, 0) + c_closed * closed_days + noise

closed_days is the number of days the market was shut between the two observations (2 for a weekend). Vendor vols
use a calendar-day clock, so a fixed-strike vol drifts down on ordinary days and jumps after weekends; c_closed absorbs
that, and `a` becomes the drift per trading day. Out of sample this term raised R-squared on 8 of 9 names
(mean 10.0% -> 13.4%). Set closed_term=False for the version without it.

Unlike the Markov model there is no regime to recognise first: a change in behaviour starts moving the betas the
next day, and old behaviour fades out on its own. The price is noise, since few observations carry most of the
weight. Two things keep that in check:

  * shrinkage: the estimate is pulled toward zero as if `prior_obs` extra observations showed no effect;
  * the outlier cap, applied using only data available at the time.

AdaptiveEwmModel runs several half-lives side by side and each day uses the one whose recent forecasts were best.
"""
import numpy as np

from .base import WINSOR, Model, design, forecast_from, path_frame

MIN_OBS = 30


def causal_winsorize(y, k=WINSOR, min_obs=MIN_OBS):
    """Cap each vol change using the median and spread of the observations before it."""
    out = np.array(y, float)
    for t in range(min_obs, len(y)):
        med = np.median(y[:t])
        mad = 1.4826 * np.median(np.abs(y[:t] - med))
        out[t] = np.clip(y[t], med - k * mad, med + k * mad)
    return out


def ewm_path(y, X, halflife, prior_obs):
    """Coefficients through time. Row t is fitted on observations before t; the last row (index len(y)) is the
    forecast for the next, unseen observation. Returns (coef, resid_sd), both with len(y) + 1 rows."""
    yc = causal_winsorize(np.asarray(y, float))
    lam = 0.5 ** (1 / halflife)
    T, k = X.shape
    coef, sd = np.full((T + 1, k), np.nan), np.full(T + 1, np.nan)
    A, b, W, S = np.zeros((k, k)), np.zeros(k), 0.0, 0.0      # weighted X'X, X'y, sum of weights, sum of squared errors
    for t in range(T + 1):
        if t >= MIN_OBS:
            shrink = prior_obs * np.diag(np.diag(A)) / W      # `prior_obs` average observations saying "no effect"
            coef[t] = np.linalg.solve(A + shrink + 1e-10 * np.eye(k), b)
            sd[t] = np.sqrt(S / W)
        if t < T:
            if t >= MIN_OBS:
                S = lam * S + (yc[t] - X[t] @ coef[t]) ** 2
            else:
                S = lam * S + yc[t] ** 2
            A = lam * A + np.outer(X[t], X[t])
            b = lam * b + X[t] * yc[t]
            W = lam * W + 1
    return coef, sd


class EwmModel(Model):
    """symmetric=True fits one beta for rallies and sell-offs alike (columns: intercept, r, closed days); the
    forecast then reports that beta as both b_up and b_dn."""

    def __init__(self, halflife=20, prior_obs=5, closed_term=True, symmetric=False):
        self.halflife, self.prior_obs, self.closed_term, self.symmetric = halflife, prior_obs, closed_term, symmetric
        self.name = f"ewm{halflife}" + ("" if closed_term else "_nowkd") + ("_sym" if symmetric else "")
        self.label = (f"Short memory, half-life {halflife} observations" + ("" if closed_term else ", no weekend term")
                      + (", one β for both directions" if symmetric else ""))

    def _X(self, r, closed):
        X = design(r, closed if self.closed_term and closed is not None else None)
        if self.symmetric:
            X = np.column_stack([X[:, 0], X[:, 1] + X[:, 2], X[:, 3:]])      # up part + down part = r
        return X

    def _expand(self, coef):
        """Symmetric coefficients (a, b, [c]) -> the common layout (a, b_up, b_dn, [c])."""
        return np.column_stack([coef[:, :2], coef[:, 1:]]) if self.symmetric else coef

    def forecast(self, y, r, closed=None):
        coef, sd = ewm_path(y, self._X(r, closed), self.halflife, self.prior_obs)
        return forecast_from(self._expand(coef[-1:])[0], sd[-1], {"halflife": self.halflife})

    def walk_forward(self, y, r, start, step=1, closed=None):
        coef, _ = ewm_path(y, self._X(r, closed), self.halflife, self.prior_obs)
        coef = self._expand(coef[:-1]).copy()
        coef[:start] = np.nan
        return path_frame(coef, r, closed if self.closed_term else None)


class AdaptiveEwmModel(Model):
    """Each day, use the half-life whose one-day-ahead forecasts had the smallest squared error recently
    (errors themselves weighted with a `judge_halflife` memory). All of it uses past data only."""
    name = "ewm_adaptive"
    label = "Short memory, half-life chosen daily"

    def __init__(self, halflives=(5, 10, 20, 40, 80), prior_obs=5, judge_halflife=40):
        self.halflives, self.prior_obs, self.judge_halflife = halflives, prior_obs, judge_halflife

    def _paths(self, y, r, closed):
        X, yc = design(r, closed), causal_winsorize(np.asarray(y, float))
        T = len(yc)
        coefs = [ewm_path(y, X, h, self.prior_obs) for h in self.halflives]
        # running score of each half-life: discounted squared forecast error up to, not including, each day
        lam = 0.5 ** (1 / self.judge_halflife)
        score = np.zeros((T + 1, len(self.halflives)))
        for i, (c, _) in enumerate(coefs):
            err2 = np.nan_to_num((yc - (c[:-1] * X).sum(axis=1)) ** 2)
            for t in range(T):
                score[t + 1, i] = lam * score[t, i] + err2[t]
        best = score.argmin(axis=1)
        coef = np.array([coefs[best[t]][0][t] for t in range(T + 1)])
        sd = np.array([coefs[best[t]][1][t] for t in range(T + 1)])
        return coef, sd, np.array(self.halflives)[best]

    def forecast(self, y, r, closed=None):
        coef, sd, chosen = self._paths(y, r, closed)
        return forecast_from(coef[-1], sd[-1], {"halflife": int(chosen[-1])})

    def walk_forward(self, y, r, start, step=1, closed=None):
        coef, _, chosen = self._paths(y, r, closed)
        coef = coef[:-1].copy()
        coef[:start] = np.nan
        return path_frame(coef, r, closed).assign(halflife=chosen[:-1])
