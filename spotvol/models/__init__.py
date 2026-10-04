"""Models of the asymmetric spot/vol beta. One file per model, all with the same interface (see base.py).

Every model explains the day's vol change with the day's spot return, split into its up and down parts:

    dvol = a + b_up * max(r, 0) + b_dn * min(r, 0) + noise

plus, except in the Markov model, c_closed * closed_days: the jump in calendar-clock vendor vols after weekends.
They differ in how the coefficients are allowed to change through time:

    markov.py   MarkovModel       two hidden regimes, each with its own betas; long memory, slow to turn
    ewm.py      EwmModel          one set of betas, recent observations weighted more (fixed half-life);
                                  also run with one beta for both directions, and without the weekend term
    ewm.py      AdaptiveEwmModel  the same, choosing its half-life each day from recent forecast errors
    crash.py    CrashEwmModel     short-memory betas plus a crash term (drops beyond 1.5 implied daily s.d.), one-year memory
    ols.py      RollingOlsModel   plain regression on the last N observations (benchmark)
    ols.py      ExpandingOlsModel plain regression on all history (benchmark)

To add a model: subclass base.Model in a new file, implement forecast(), and add it to MODELS below.
"""
from .base import Forecast, Model, design, winsorize
from .crash import CrashEwmModel
from .ewm import AdaptiveEwmModel, EwmModel
from .markov import MarkovModel
from .ols import ExpandingOlsModel, RollingOlsModel

# the models shown in the monitor, the screen and the backtest, in display order
MODELS = [EwmModel(halflife=20), CrashEwmModel(k=1.5, scale="implied"), EwmModel(halflife=20, symmetric=True), EwmModel(halflife=20, closed_term=False),
          AdaptiveEwmModel(), MarkovModel(),
          RollingOlsModel(60), ExpandingOlsModel()]
PRIMARY = "ewm20"      # the model whose betas head the screen


def get(name):
    return next(m for m in MODELS if m.name == name)
