"""Regime-switching, asymmetric spot/vol beta.

Model, per observation t (same pairs as analysis.py, earnings excluded):

    dvol_t = a_s + b_up_s * max(r_t, 0) + b_dn_s * min(r_t, 0) + e_t,   e_t ~ N(0, sig_s^2)

where s is a hidden Markov regime with transition matrix P. Fitted by EM (Hamilton filter + Kim smoother,
weighted least squares in the M-step). The residual variance is shared across regimes, so regimes have to
differ in their betas rather than in how noisy the day was, and dvol is winsorized at 4 robust standard
deviations before fitting. Of the variants tried, this one was the most stable out of sample across names. b_up and b_dn are both slopes in vol points per 1% of spot return:
b_up > 0 means vol rises on rallies; b_dn < 0 means vol rises on sell-offs.

The filtered probabilities use data up to each date only, so "today's regime" and the next-day betas are
usable in real time. walk_forward() checks that out of sample.
"""
import numpy as np
import pandas as pd

from . import analysis as an

MAX_K = 2
WINSOR = 4.0
MIN_REGIME_OBS = 25
SCENARIOS = [-5, -3, -1, 1, 3, 5]


def design(r):
    r = np.asarray(r, float)
    return np.column_stack([np.ones_like(r), np.clip(r, 0, None), np.clip(r, None, 0)])


def _emission(y, X, theta, sig):
    res = y[:, None] - X @ theta.T
    return np.exp(-0.5 * (res / sig) ** 2) / (sig * np.sqrt(2 * np.pi)) + 1e-300


def _filter(B, P, pi):
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
    T, K = B.shape
    beta = np.ones((T, K))
    xi = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        nb = B[t + 1] * beta[t + 1] / c[t + 1]
        beta[t] = P @ nb
        xi += alpha[t][:, None] * P * nb[None, :]
    return alpha * beta, xi


def _m_step(y, X, gamma, floor, shared_var=True):
    K = gamma.shape[1]
    theta, sig = np.empty((K, X.shape[1])), np.empty(K)
    for k in range(K):
        w = gamma[:, k]
        XtW = X.T * w
        theta[k] = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ y)
        sig[k] = max(np.sqrt(w @ (y - X @ theta[k]) ** 2 / w.sum()), floor)
    if shared_var:
        sig[:] = np.sqrt((gamma.sum(axis=0) @ sig ** 2) / len(y))
    return theta, sig


def _em(y, X, theta, sig, P, pi, shared_var=True, max_iter=300, tol=1e-6):
    floor = 0.05 * y.std()
    ll_old = -np.inf
    for _ in range(max_iter):
        B = _emission(y, X, theta, sig)
        alpha, c = _filter(B, P, pi)
        ll = np.log(c).sum()
        gamma, xi = _smooth(B, P, alpha, c)
        theta, sig = _m_step(y, X, gamma, floor, shared_var)
        P = xi / xi.sum(axis=1, keepdims=True)
        pi = gamma[0]
        if ll - ll_old < tol:
            break
        ll_old = ll
    return theta, sig, P, pi


def fit(y, X, K, init=None, restarts=6, seed=0, shared_var=True):
    """Fit a K-regime model. init = (theta, sig, P, pi) warm-starts a single run."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    T = len(y)
    rng = np.random.default_rng(seed)
    # a warm start alone can stay stuck in yesterday's solution, so `restarts` random starts compete with it
    starts = ([init] if init is not None else []) + [None] * (1 if K == 1 else restarts)
    best = None
    for s in starts:
        if s is None:
            # random start: regimes assigned in blocks of 40 observations
            lab = np.repeat(rng.integers(0, K, T // 40 + 1), 40)[:T]
            gamma = np.full((T, K), 0.1 / K) + 0.9 * np.eye(K)[lab]
            theta, sig = _m_step(y, X, gamma, 0.05 * y.std(), shared_var)
            P = np.full((K, K), 0.05 / max(K - 1, 1)) + (0.95 - 0.05 / max(K - 1, 1)) * np.eye(K) if K > 1 else np.ones((1, 1))
            s = (theta, sig, P, np.full(K, 1 / K))
        theta, sig, P, pi = _em(y, X, *s, shared_var=shared_var)
        B = _emission(y, X, theta, sig)
        alpha, c = _filter(B, P, pi)
        ll = np.log(c).sum()
        if best is None or ll > best["loglik"]:
            best = {"theta": theta, "sig": sig, "P": P, "pi": pi, "loglik": ll, "filt": alpha, "smooth": _smooth(B, P, alpha, c)[0]}
    # order regimes by up-beta so labels are stable from one day's refit to the next
    o = np.argsort(best["theta"][:, 1])
    f = {"theta": best["theta"][o], "sig": best["sig"][o], "P": best["P"][np.ix_(o, o)], "pi": best["pi"][o],
         "loglik": best["loglik"], "filt": best["filt"][:, o], "smooth": best["smooth"][:, o], "K": K}
    n_par = K * X.shape[1] + (1 if shared_var else K) + K * (K - 1) + (K - 1)
    f["bic"] = -2 * f["loglik"] + n_par * np.log(T)
    f["n_eff"] = f["smooth"].sum(axis=0)
    f["se"] = np.array([np.sqrt(np.diag(f["sig"][k] ** 2 * np.linalg.inv((X.T * f["smooth"][:, k]) @ X + 1e-8 * np.eye(X.shape[1]))))
                        for k in range(K)])
    return f


def winsorize(y, k=WINSOR):
    med = np.median(y)
    mad = 1.4826 * np.median(np.abs(y - med))
    return np.clip(y, med - k * mad, med + k * mad)


def select(y, X):
    """One regime or two, by BIC, rejecting a fit with a regime too thin to trust."""
    fits = [fit(y, X, k) for k in range(1, MAX_K + 1)]
    ok = [f for f in fits if f["n_eff"].min() >= MIN_REGIME_OBS]
    return min(ok, key=lambda f: f["bic"]), {f["K"]: f["bic"] for f in fits}


def _name(b_up, se_up, b_dn, se_dn):
    up = "up" if b_up > 2 * se_up else "down" if b_up < -2 * se_up else "flat"
    dn = "up" if b_dn < -2 * se_dn else "down" if b_dn > 2 * se_dn else "flat"      # vol direction on a sell-off
    return {("up", "up"): "V-shape: vol up on rallies and on sell-offs",
            ("down", "up"): "Normal: vol down on rallies, up on sell-offs",
            ("up", "down"): "Inverted: vol up on rallies, down on sell-offs",
            ("up", "flat"): "Vol up on rallies only", ("flat", "up"): "Vol up on sell-offs only",
            ("down", "flat"): "Vol down on rallies only", ("flat", "down"): "Vol down on sell-offs only",
            ("down", "down"): "Vol down on any move", ("flat", "flat"): "No spot/vol link"}[(up, dn)]


def walk_forward(y, X, K, start=250, step=10):
    """Out-of-sample check: refit every `step` observations on data so far, then predict each next observation's
    vol change given its spot move. R-squared is against predicting no vol change, on the raw (unwinsorized) data."""
    T = len(y)
    if T < start + 30:
        return None
    pred = {"regime": np.full(T, np.nan), "single": np.full(T, np.nan), "rolling60": np.full(T, np.nan)}
    f = None
    for t0 in range(start, T, step):
        t1 = min(t0 + step, T)
        yw = winsorize(y[:t0])
        f = fit(yw, X[:t0], K, init=(f["theta"], f["sig"], f["P"], f["pi"]) if f else None, restarts=6 if f is None else 0)
        yf = np.clip(y[:t1], yw.min(), yw.max())
        alpha, _ = _filter(_emission(yf, X[:t1], f["theta"], f["sig"]), f["P"], f["pi"])
        ols = np.linalg.lstsq(X[:t0], yw, rcond=None)[0]
        roll = np.linalg.lstsq(X[t0 - 60:t0], yw[-60:], rcond=None)[0]
        for t in range(t0, t1):
            p = alpha[t - 1] @ f["P"]                     # regime probabilities for t, from data up to t-1
            pred["regime"][t] = p @ (f["theta"] @ X[t])
            pred["single"][t] = X[t] @ ols
            pred["rolling60"][t] = X[t] @ roll
    m = ~np.isnan(pred["regime"])
    up, dn = m & (X[:, 1] > 0), m & (X[:, 2] < 0)

    def r2(v, q):
        return 1 - ((y[q] - v[q]) ** 2).sum() / (y[q] ** 2).sum()

    labels = {"regime": f"Regime-switching ({K} regimes)" if K > 1 else "Single regime (model selected)",
              "single": "Single regime, all history", "rolling60": "Rolling 60 observations"}
    return pd.DataFrame([{"model": labels[k], "n": int(m.sum()), "oos_r2": r2(v, m), "oos_r2_up": r2(v, up), "oos_r2_down": r2(v, dn)}
                         for k, v in pred.items()])


def monitor(symbol, tenor="1m", target="dFix", with_oos=False):
    """Fit on all history and describe today's regime and the next-day asymmetric betas."""
    res = an.analyse(symbol)
    if res is None or tenor not in res["tenors"]:
        return None
    p = res["tenors"][tenor]["pairs"]
    if len(p) < 120:
        return None
    y, X = p[target].to_numpy(), design(p["r"])
    f, bics = select(winsorize(y), X)
    K = f["K"]
    now, nxt = f["filt"][-1], f["filt"][-1] @ f["P"]
    regimes = pd.DataFrame([{
        "regime": f"{k + 1}. " + _name(f["theta"][k, 1], f["se"][k, 1], f["theta"][k, 2], f["se"][k, 2]),
        "b_up": f["theta"][k, 1], "t_up": f["theta"][k, 1] / f["se"][k, 1], "b_dn": f["theta"][k, 2], "t_dn": f["theta"][k, 2] / f["se"][k, 2],
        "share": f["smooth"][:, k].mean(), "stay_prob": f["P"][k, k],
        "duration": 1 / max(1 - f["P"][k, k], 1e-9), "prob_now": now[k], "prob_next": nxt[k]} for k in range(K)])
    scen = []
    for r in SCENARIOS:
        mu = f["theta"] @ design([r])[0]
        mean = nxt @ mu
        scen.append({"move": f"{r:+d}%", "exp_dvol": mean, "sd_dvol": np.sqrt(nxt @ (f["sig"] ** 2 + mu ** 2) - mean ** 2)})
    probs = pd.DataFrame(f["smooth"], columns=list(regimes["regime"])).assign(date=p["d1"].to_numpy())
    k_now = int(now.argmax())
    roll = np.linalg.lstsq(X[-60:], winsorize(y)[-60:], rcond=None)[0]
    out = {"symbol": symbol, "tenor": tenor, "target": target, "date": p["d1"].iloc[-1], "K": K, "bic": bics, "n": len(p),
           "regimes": regimes, "scenarios": pd.DataFrame(scen), "probs": probs,
           "regime_name": regimes["regime"][k_now], "prob_now": now[k_now],
           "b_up_next": nxt @ f["theta"][:, 1], "b_dn_next": nxt @ f["theta"][:, 2],
           "b_up_roll60": roll[1], "b_dn_roll60": roll[2]}
    if with_oos:
        out["oos"] = walk_forward(y, X, K)
    return out
