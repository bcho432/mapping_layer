"""Exponential smoothing — SES, Holt's linear, and Holt-Winters additive.

Pure standard library. No statsmodels, no numpy, nothing to install.

This exists because a dummy engine that echoes its input proves the plumbing
works and nothing else. ETS produces a real forecast with real error, so
"the pipeline works" becomes a number you can check rather than a claim.

The three models are the same recursion with pieces switched off:

    SES            level only
    Holt           level + trend
    Holt-Winters   level + trend + season          (additive)

Parameters are fitted by grid search on in-sample SSE. That is cruder than the
L-BFGS statsmodels uses, but it is deterministic, dependency-free, and on
series of a few hundred points the difference is not what will decide whether
this architecture is sound.
"""

import re
from datetime import date

_GRAIN_PATTERNS = (
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), "day"),
    (re.compile(r"^(\d{4})-W(\d{2})$"),        "week"),
    (re.compile(r"^(\d{4})-(\d{2})$"),         "month"),
    (re.compile(r"^(\d{4})-Q(\d)$"),           "quarter"),
    (re.compile(r"^(\d{4})$"),                  "year"),
)


def detect_grain(label):
    for pat, grain in _GRAIN_PATTERNS:
        if pat.match(str(label)):
            return grain
    return None


def period_index(label):
    """A bucket label as a monotonic integer, so gaps are detectable."""
    g, s = detect_grain(label), str(label)
    try:
        if g == "month":
            y, m = s.split("-"); return int(y) * 12 + int(m)
        if g == "quarter":
            y, q = s.split("-Q"); return int(y) * 4 + int(q)
        if g == "year":
            return int(s)
        if g == "week":
            y, w = s.split("-W"); return int(y) * 53 + int(w)
        if g == "day":
            return date.fromisoformat(s).toordinal()
    except (ValueError, TypeError):
        return None
    return None

GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _fit_once(y, alpha, beta, gamma, m):
    """Run the recursion. Returns (fitted, level, trend, season) or None."""
    n = len(y)
    if m and n < 2 * m:
        return None

    if m:
        seasons = [y[i:i + m] for i in range(0, n - n % m, m)]
        if len(seasons) < 2:
            return None
        a0 = sum(seasons[0]) / m
        a1 = sum(seasons[1]) / m
        level = a0
        trend = (a1 - a0) / m if beta is not None else 0.0
        season = [y[i] - a0 for i in range(m)]
    else:
        level = y[0]
        trend = (y[1] - y[0]) if (beta is not None and n > 1) else 0.0
        season = None

    fitted = []
    for t in range(n):
        s_idx = t % m if m else None
        s_prev = season[s_idx] if m else 0.0
        pred = level + trend + s_prev
        fitted.append(pred)

        err_base = y[t] - s_prev
        last_level = level
        level = alpha * err_base + (1 - alpha) * (level + trend)
        if beta is not None:
            trend = beta * (level - last_level) + (1 - beta) * trend
        if m and gamma is not None:
            season[s_idx] = (gamma * (y[t] - last_level - trend)
                             + (1 - gamma) * s_prev)

    return fitted, level, trend, season


def _sse(y, fitted):
    return sum((a - b) ** 2 for a, b in zip(y, fitted))


def fit(y, season=None):
    """Pick the best of SES / Holt / Holt-Winters by in-sample SSE."""
    y = [float(v) for v in y]
    best = None
    m = season if (season and len(y) >= 2 * season) else None

    combos = []
    for a in GRID:
        combos.append((a, None, None, None))                 # SES
        for b in GRID:
            combos.append((a, b, None, None))                # Holt
            if m:
                for g in GRID:
                    combos.append((a, b, g, m))              # Holt-Winters

    for alpha, beta, gamma, mm in combos:
        out = _fit_once(y, alpha, beta, gamma, mm)
        if out is None:
            continue
        score = _sse(y, out[0])
        if best is None or score < best["sse"]:
            best = {"sse": score, "alpha": alpha, "beta": beta, "gamma": gamma,
                    "m": mm, "fitted": out[0], "level": out[1],
                    "trend": out[2], "season": out[3]}
    return best


def forecast(state, h):
    """h steps ahead from a fitted state."""
    level, trend, season, m = (state["level"], state["trend"],
                               state["season"], state["m"])
    n = len(state["fitted"])
    out = []
    for step in range(1, h + 1):
        s = season[(n + step - 1) % m] if m else 0.0
        out.append(level + step * trend + s)
    return out


# ------------------------------------------------------------------ accuracy

def mape(actual, pred):
    pairs = [(a, p) for a, p in zip(actual, pred) if a not in (0, None)]
    if not pairs:
        return None
    return 100 * sum(abs((a - p) / a) for a, p in pairs) / len(pairs)


def mae(actual, pred):
    return sum(abs(a - p) for a, p in zip(actual, pred)) / len(actual) if actual else None


def mape_is_meaningful(actual):
    """Whether a percentage error can be trusted on this series.

    MAPE divides by the actual value. On federal spend, a quarter where an
    industry obligated almost nothing turns a small absolute miss into a
    percentage in the millions — 138,900,653% was a real number this produced.
    That is not a large error, it is a broken statistic.

    Call it unusable when any actual is negligible next to the series' own
    typical size, or when anything is negative (percentages of negatives are
    meaningless too).
    """
    vals = [a for a in actual if a is not None]
    if not vals:
        return False
    if any(v < 0 for v in vals):
        return False
    mags = sorted(abs(v) for v in vals)
    typical = mags[len(mags) // 2]
    if typical == 0:
        return False
    return mags[0] >= 0.01 * typical


def mase(actual, pred, train, season=None):
    """Mean Absolute Scaled Error — MAE divided by the in-sample naive MAE.

    Scale-free like MAPE but defined at zero, which is why it is the right
    headline for spiky data. Below 1.0 means the model beats the naive method
    it is scaled against; above 1.0 means it does not.
    """
    if not actual or len(train) < 2:
        return None
    m = season if (season and len(train) > season) else 1
    diffs = [abs(train[i] - train[i - m]) for i in range(m, len(train))]
    scale = sum(diffs) / len(diffs) if diffs else None
    if not scale:
        return None
    return mae(actual, pred) / scale


def rmse(actual, pred):
    if not actual:
        return None
    return (sum((a - p) ** 2 for a, p in zip(actual, pred)) / len(actual)) ** 0.5


def evaluate(y, season=None, holdout=None):
    """Honest accuracy: fit on the front, score on a held-out tail.

    Compared against a NAIVE baseline (carry the last value forward), because
    an error number on its own says nothing — the question is whether the model
    beat the thing you get for free.
    """
    y = [float(v) for v in y]
    h = holdout or max(1, min(len(y) // 4, season or 4))
    if len(y) < h + 4:
        return None
    train, test = y[:-h], y[-h:]

    state = fit(train, season)
    if not state:
        return None
    pred = forecast(state, h)
    naive = [train[-1]] * h

    r_model, r_naive = rmse(test, pred), rmse(test, naive)

    # A perfectly flat series makes the naive baseline exact, so its RMSE is 0
    # and the ratio is undefined. Returning None here used to crash the caller,
    # which summed these without checking. Say 0 when both are exact (a genuine
    # tie) and -100 when the model is worse than a perfect baseline; reserve
    # None for the case where there is truly nothing to compare.
    if r_naive is None or r_model is None:
        skill = None
    elif r_naive == 0:
        skill = 0.0 if r_model == 0 else -100.0
    else:
        skill = (1 - (r_model / r_naive)) * 100

    usable = mape_is_meaningful(test)
    out = {
        "method": "holdout",
        "holdout": h,
        "train_rows": len(train),
        "model": ("holt-winters" if state["m"] else
                  "holt" if state["beta"] is not None else "ses"),
        "params": {k: state[k] for k in ("alpha", "beta", "gamma", "m")},
        "mae": _r(mae(test, pred)),
        "rmse": _r(r_model),
        "mase": _r(mase(test, pred, train, state["m"])),
        "naive_mae": _r(mae(test, naive)),
        "naive_rmse": _r(r_naive),
        "skill_vs_naive": _r(skill),
        "mape_usable": usable,
    }
    if usable:
        out["mape"] = _r(mape(test, pred))
        out["naive_mape"] = _r(mape(test, naive))
    return out


def evaluate_rolling(y, season=None, horizon=1, max_folds=24):
    """Walk-forward evaluation: many origins instead of one.

    A single holdout of size h scores the forecast on exactly h points, so a
    2-period horizon rests its whole accuracy estimate on 2 numbers. That is
    why MAPE bounced between 3.0 and 5.1 across horizons on the same series —
    noise, not skill.

    Here the origin slides: fit on y[:i], predict h ahead, score against
    y[i:i+h], move forward, repeat. Errors from every fold are pooled, so a
    2-period horizon can still be judged on dozens of predictions.

    Parameters are re-tuned at every origin, on that origin's training data
    only. Tuning once on the first window and holding it fixed was tried first
    and is badly pessimistic: on the airline series the first 24 months pick
    alpha=0.9/gamma=0.1 while the full series wants alpha=0.3/gamma=0.9, and
    using the early parameters throughout dropped apparent skill from 63% to
    3%. Re-tuning costs about 0.7s for 24 folds and leaks nothing, because no
    fold ever sees data past its own origin.
    """
    y = [float(v) for v in y]
    h = max(1, int(horizon))
    m = season if (season and len(y) >= 2 * season) else None
    min_train = max(4, 2 * m if m else 4)
    if len(y) < min_train + h:
        return None

    origins = list(range(min_train, len(y) - h + 1))
    if not origins:
        return None
    if len(origins) > max_folds:                      # spread them out evenly
        step = len(origins) / max_folds
        origins = [origins[int(i * step)] for i in range(max_folds)]

    act, pred, naive_pred, last = [], [], [], None
    for i in origins:
        train, test = y[:i], y[i:i + h]
        state = fit(train, season)          # re-tuned on THIS origin only
        if not state:
            continue
        last = state
        act.extend(test)
        pred.extend(forecast(state, h))
        naive_pred.extend([train[-1]] * h)

    if not act or last is None:
        return None
    alpha, beta, gamma, mm = (last["alpha"], last["beta"],
                              last["gamma"], last["m"])
    r_model, r_naive = rmse(act, pred), rmse(act, naive_pred)
    if r_naive is None or r_model is None:
        skill = None
    elif r_naive == 0:
        skill = 0.0 if r_model == 0 else -100.0
    else:
        skill = (1 - r_model / r_naive) * 100

    usable = mape_is_meaningful(act)
    out = {
        "method": "rolling-origin",
        "folds": len(origins),
        "points_scored": len(act),
        "holdout": h,
        "train_rows": origins[0],
        "model": ("holt-winters" if mm else
                  "holt" if beta is not None else "ses"),
        "params": {"alpha": alpha, "beta": beta, "gamma": gamma, "m": mm},
        "mae": _r(mae(act, pred)),
        "rmse": _r(r_model),
        "mase": _r(mase(act, pred, y[:origins[-1]], m)),
        "naive_mae": _r(mae(act, naive_pred)),
        "naive_rmse": _r(r_naive),
        "skill_vs_naive": _r(skill),
        "mape_usable": usable,
    }
    if usable:
        out["mape"] = _r(mape(act, pred))
        out["naive_mape"] = _r(mape(act, naive_pred))
    else:
        out["mape_suppressed"] = ("values at or near zero make a percentage "
                                  "error meaningless — use MAE or MASE")
    return out


def _r(v, nd=3):
    return round(v, nd) if isinstance(v, (int, float)) else v


# ------------------------------------------------------- period continuation

def next_periods(last, h):
    """Continue a bucket label forward. Falls back to `+1, +2, …` if unknown."""
    grain = detect_grain(last)
    out = []
    if grain == "month":
        y, mth = (int(x) for x in str(last).split("-"))
        for _ in range(h):
            mth += 1
            if mth > 12:
                mth, y = 1, y + 1
            out.append(f"{y}-{mth:02d}")
    elif grain == "quarter":
        y, q = str(last).split("-Q")
        y, q = int(y), int(q)
        for _ in range(h):
            q += 1
            if q > 4:
                q, y = 1, y + 1
            out.append(f"{y}-Q{q}")
    elif grain == "year":
        y = int(last)
        out = [str(y + i) for i in range(1, h + 1)]
    elif grain == "day":
        from datetime import date, timedelta
        d = date.fromisoformat(str(last))
        out = [(d + timedelta(days=i)).isoformat() for i in range(1, h + 1)]
    elif grain == "week":
        y, w = str(last).split("-W")
        y, w = int(y), int(w)
        for _ in range(h):
            w += 1
            if w > 52:
                w, y = 1, y + 1
            out.append(f"{y}-W{w:02d}")
    else:
        out = [f"{last}+{i}" for i in range(1, h + 1)]
    return out
