"""Short-memory model: exponentially weighted regression.

One set of betas, re-estimated every day, with an observation's weight halving every `halflife` observations:

    weight of an observation k days old = 0.5 ** (k / halflife)

Unlike the Markov model there is no regime to recognise first: a change in behaviour starts moving the betas the
next day, and old behaviour fades out on its own. The price is noise, since few observations carry most of the
weight. Two things keep that in check:

  * shrinkage: the estimate is pulled toward zero as if `prior_obs` extra observations showed no spot/vol link;
  * the outlier cap, applied using only data available at the time.

AdaptiveEwmModel runs several half-lives side by side and each day uses the one whose recent forecasts were best.
"""
import numpy as np

from .base import WINSOR, Forecast, Model, design, path_frame

MIN_OBS = 30


def causal_winsorize(y, k=WINSOR, min_obs=MIN_OBS):
    """Cap each vol change using the median and spread of the observations before it."""
    out = np.array(y, float)
    for t in range(min_obs, len(y)):
        med = np.median(y[:t])
        mad = 1.4826 * np.median(np.abs(y[:t] - med))
        out[t] = np.clip(y[t], med - k * mad, med + k * mad)
    return out


def ewm_path(y, r, halflife, prior_obs):
    """Coefficients through time. Row t is fitted on observations before t; the last row (index len(y)) is the
    forecast for the next, unseen observation. Returns (coef, resid_sd), both with len(y) + 1 rows."""
    X, yc = design(r), causal_winsorize(np.asarray(y, float))
    lam = 0.5 ** (1 / halflife)
    T = len(yc)
    coef, sd = np.full((T + 1, 3), np.nan), np.full(T + 1, np.nan)
    A, b, W, S = np.zeros((3, 3)), np.zeros(3), 0.0, 0.0      # weighted X'X, X'y, sum of weights, sum of squared errors
    for t in range(T + 1):
        if t >= MIN_OBS:
            shrink = prior_obs * np.diag(np.diag(A)) / W      # `prior_obs` average observations saying "no link"
            coef[t] = np.linalg.solve(A + shrink, b)
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
    def __init__(self, halflife=20, prior_obs=5):
        self.halflife, self.prior_obs = halflife, prior_obs
        self.name = f"ewm{halflife}"
        self.label = f"Short memory, half-life {halflife} observations"

    def forecast(self, y, r):
        coef, sd = ewm_path(y, r, self.halflife, self.prior_obs)
        return Forecast(*coef[-1], resid_sd=float(sd[-1]), detail={"halflife": self.halflife})

    def walk_forward(self, y, r, start, step=1):
        coef, _ = ewm_path(y, r, self.halflife, self.prior_obs)
        coef = coef[:-1].copy()
        coef[:start] = np.nan
        return path_frame(coef, r)


class AdaptiveEwmModel(Model):
    """Each day, use the half-life whose one-day-ahead forecasts had the smallest squared error recently
    (errors themselves weighted with a `judge_halflife` memory). All of it uses past data only."""
    name = "ewm_adaptive"
    label = "Short memory, half-life chosen daily"

    def __init__(self, halflives=(5, 10, 20, 40, 80), prior_obs=5, judge_halflife=40):
        self.halflives, self.prior_obs, self.judge_halflife = halflives, prior_obs, judge_halflife

    def _paths(self, y, r):
        X, yc = design(r), causal_winsorize(np.asarray(y, float))
        T = len(yc)
        coefs = [ewm_path(y, r, h, self.prior_obs) for h in self.halflives]
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

    def forecast(self, y, r):
        coef, sd, chosen = self._paths(y, r)
        return Forecast(*coef[-1], resid_sd=float(sd[-1]), detail={"halflife": int(chosen[-1])})

    def walk_forward(self, y, r, start, step=1):
        coef, _, chosen = self._paths(y, r)
        coef = coef[:-1].copy()
        coef[:start] = np.nan
        return path_frame(coef, r).assign(halflife=chosen[:-1])
