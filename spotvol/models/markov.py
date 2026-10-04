"""Regime-switching model: a hidden Markov chain picks which set of betas applies each day.

    dvol_t = a_s + b_up_s * max(r_t, 0) + b_dn_s * min(r_t, 0) + e_t,   e_t ~ N(0, sig^2),   s_t in {1..K}

Fitted by EM (Hamilton filter + Kim smoother, weighted least squares in the M-step). The residual variance is shared
across regimes, so regimes have to differ in their betas rather than in how noisy the day was, and dvol is
winsorized before fitting. The filtered probabilities use data up to each date only, so they are usable in real time.

Long memory: the betas of each regime are estimated from all history, and a regime change is only recognised
after enough contrary days. The backtest shows this makes it slow to turn; see ewm.py for the short-memory model.

It does not use the closed-day (weekend) term that the other models have; the calendar-clock jump after weekends
therefore sits in its noise.
"""
import numpy as np
import pandas as pd

from .base import Forecast, Model, design, path_frame, winsorize

MIN_REGIME_OBS = 25


# ---------- EM machinery ----------

def _emission(y, X, theta, sig):
    res = y[:, None] - X @ theta.T
    return np.exp(-0.5 * (res / sig) ** 2) / (sig * np.sqrt(2 * np.pi)) + 1e-300


def _filter(B, P, pi):
    """Hamilton filter: alpha[t] = P(regime at t | data up to t)."""
    T, K = B.shape
    alpha, c = np.empty((T, K)), np.empty(T)
    a = pi * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ P) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c


def _smooth(B, P, alpha, c):
    """Kim smoother: P(regime at t | all data), and the expected transition counts."""
    T, K = B.shape
    beta = np.ones((T, K))
    xi = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        nb = B[t + 1] * beta[t + 1] / c[t + 1]
        beta[t] = P @ nb
        xi += alpha[t][:, None] * P * nb[None, :]
    return alpha * beta, xi


def _m_step(y, X, gamma, floor):
    K = gamma.shape[1]
    theta, sig = np.empty((K, X.shape[1])), np.empty(K)
    for k in range(K):
        w = gamma[:, k]
        XtW = X.T * w
        theta[k] = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ y)
        sig[k] = max(np.sqrt(w @ (y - X @ theta[k]) ** 2 / w.sum()), floor)
    sig[:] = np.sqrt((gamma.sum(axis=0) @ sig ** 2) / len(y))       # shared variance
    return theta, sig


def _em(y, X, theta, sig, P, pi, max_iter=300, tol=1e-6):
    floor = 0.05 * y.std()
    ll_old = -np.inf
    for _ in range(max_iter):
        B = _emission(y, X, theta, sig)
        alpha, c = _filter(B, P, pi)
        ll = np.log(c).sum()
        gamma, xi = _smooth(B, P, alpha, c)
        theta, sig = _m_step(y, X, gamma, floor)
        P = xi / xi.sum(axis=1, keepdims=True)
        pi = gamma[0]
        if ll - ll_old < tol:
            break
        ll_old = ll
    return theta, sig, P, pi


def fit(y, X, K, init=None, restarts=6, seed=0):
    """Fit a K-regime model. init = (theta, sig, P, pi) adds a warm start to the `restarts` random ones."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    T = len(y)
    rng = np.random.default_rng(seed)
    # a warm start alone can stay stuck in yesterday's solution, so random starts compete with it
    starts = ([init] if init is not None else []) + [None] * (1 if K == 1 else restarts)
    best = None
    for s in starts:
        if s is None:
            # random start: regimes assigned in blocks of 40 observations
            lab = np.repeat(rng.integers(0, K, T // 40 + 1), 40)[:T]
            gamma = np.full((T, K), 0.1 / K) + 0.9 * np.eye(K)[lab]
            theta, sig = _m_step(y, X, gamma, 0.05 * y.std())
            P = np.full((K, K), 0.05 / max(K - 1, 1)) + (0.95 - 0.05 / max(K - 1, 1)) * np.eye(K) if K > 1 else np.ones((1, 1))
            s = (theta, sig, P, np.full(K, 1 / K))
        theta, sig, P, pi = _em(y, X, *s)
        B = _emission(y, X, theta, sig)
        alpha, c = _filter(B, P, pi)
        ll = np.log(c).sum()
        if best is None or ll > best["loglik"]:
            best = {"theta": theta, "sig": sig, "P": P, "pi": pi, "loglik": ll, "filt": alpha, "smooth": _smooth(B, P, alpha, c)[0]}
    # order regimes by up-beta so labels are stable from one day's refit to the next
    o = np.argsort(best["theta"][:, 1])
    f = {"theta": best["theta"][o], "sig": best["sig"][o], "P": best["P"][np.ix_(o, o)], "pi": best["pi"][o],
         "loglik": best["loglik"], "filt": best["filt"][:, o], "smooth": best["smooth"][:, o], "K": K}
    n_par = K * X.shape[1] + 1 + K * (K - 1) + (K - 1)
    f["bic"] = -2 * f["loglik"] + n_par * np.log(T)
    f["n_eff"] = f["smooth"].sum(axis=0)
    f["se"] = np.array([np.sqrt(np.diag(f["sig"][k] ** 2 * np.linalg.inv((X.T * f["smooth"][:, k]) @ X + 1e-8 * np.eye(X.shape[1]))))
                        for k in range(K)])
    return f


def select(y, X, max_k=2):
    """One regime or two, by BIC, rejecting a fit with a regime too thin to trust."""
    fits = [fit(y, X, k) for k in range(1, max_k + 1)]
    ok = [f for f in fits if f["n_eff"].min() >= MIN_REGIME_OBS]
    return min(ok, key=lambda f: f["bic"]), {f["K"]: f["bic"] for f in fits}


def regime_name(b_up, se_up, b_dn, se_dn):
    up = "up" if b_up > 2 * se_up else "down" if b_up < -2 * se_up else "flat"
    dn = "up" if b_dn < -2 * se_dn else "down" if b_dn > 2 * se_dn else "flat"      # vol direction on a sell-off
    return {("up", "up"): "V-shape: vol up on rallies and on sell-offs",
            ("down", "up"): "Normal: vol down on rallies, up on sell-offs",
            ("up", "down"): "Inverted: vol up on rallies, down on sell-offs",
            ("up", "flat"): "Vol up on rallies only", ("flat", "up"): "Vol up on sell-offs only",
            ("down", "flat"): "Vol down on rallies only", ("flat", "down"): "Vol down on sell-offs only",
            ("down", "down"): "Vol down on any move", ("flat", "flat"): "No spot/vol link"}[(up, dn)]


def regime_table(f):
    """One row per regime of a fit: betas, how long it lasts, and its probability today and tomorrow."""
    now, nxt = f["filt"][-1], f["filt"][-1] @ f["P"]
    return pd.DataFrame([{
        "regime": f"{k + 1}. " + regime_name(f["theta"][k, 1], f["se"][k, 1], f["theta"][k, 2], f["se"][k, 2]),
        "b_up": f["theta"][k, 1], "t_up": f["theta"][k, 1] / f["se"][k, 1], "b_dn": f["theta"][k, 2], "t_dn": f["theta"][k, 2] / f["se"][k, 2],
        "share": f["smooth"][:, k].mean(), "stay_prob": f["P"][k, k], "duration": 1 / max(1 - f["P"][k, k], 1e-9),
        "prob_now": now[k], "prob_next": nxt[k]} for k in range(f["K"])])


# ---------- the model ----------

class MarkovModel(Model):
    name = "markov"
    label = "Regime-switching (Markov)"

    def __init__(self, refit_every=5):
        self.refit_every = refit_every

    def forecast(self, y, r, closed=None):
        """Number of regimes chosen by BIC. The forecast betas are the regimes' betas weighted by tomorrow's
        regime probabilities."""
        X = design(r)
        f, bics = select(winsorize(np.asarray(y, float)), X)
        nxt = f["filt"][-1] @ f["P"]
        a, b_up, b_dn = nxt @ f["theta"]
        table = regime_table(f)
        k_now = int(f["filt"][-1].argmax())
        return Forecast(a, b_up, b_dn, resid_sd=float(np.sqrt(nxt @ f["sig"] ** 2)),
                        detail={"n_regimes": f["K"], "bic": bics, "regimes": table, "smooth": f["smooth"],
                                "regime": table["regime"][k_now], "prob": float(f["filt"][-1][k_now])})

    def walk_forward(self, y, r, start, step=1, closed=None, n_regimes=2):
        """Parameters are refitted every `refit_every` observations on earlier data only; between refits the regime
        probability is still updated with each new observation. Two regimes throughout."""
        y, X = np.asarray(y, float), design(r)
        T = len(y)
        coef, p_hi = np.full((T, 3), np.nan), np.full(T, np.nan)
        f = None
        for t0 in range(start, T, self.refit_every):
            t1 = min(t0 + self.refit_every, T)
            yw = winsorize(y[:t0])
            f = fit(yw, X[:t0], n_regimes, init=(f["theta"], f["sig"], f["P"], f["pi"]) if f else None, restarts=6 if f is None else 2)
            yf = np.clip(y[:t1], yw.min(), yw.max())
            alpha, _ = _filter(_emission(yf, X[:t1], f["theta"], f["sig"]), f["P"], f["pi"])
            for t in range(t0, t1):
                p = alpha[t - 1] @ f["P"]             # regime probabilities for t, from data up to t-1
                coef[t], p_hi[t] = p @ f["theta"], p[-1]
        return path_frame(coef, r).assign(p_hi=p_hi)
