"""Item-item collaborative filtering, stdlib only.

The recommendation counterpart to ets.py. Same discipline: a real model, a real
baseline, and a score measured on interactions held back from fitting — so
"it works" is a number rather than a claim.

THE MODEL. Two items are similar when the same people engaged with both. That
is cosine similarity over the item columns of the user x item matrix. To score
a candidate for a user, add up how similar it is to everything that user has
already engaged with. No factorisation, no gradients, no dependency.

THE BASELINE. Popularity — recommend the globally most-engaged items to
everybody. This is the naive-forecast of recommendation: free, surprisingly
hard to beat, and the only honest thing to measure against. A recommender that
cannot beat popularity has not earned its place, and most published lifts look
much smaller once popularity is in the table.

THE SPLIT. Leave-k-out per user: hold back a slice of each user's interactions,
fit on the rest, then ask whether the held-out items come back in the top-k.
Users with too little history to split are excluded and counted, never quietly
scored as zero.

WHAT THIS IS NOT. Negative sampling, implicit-feedback weighting, and matrix
factorisation belong to a real library — RecBole, implicit, LightFM. This
exists so the pipeline can be judged end to end without one, and so the adapter
has a reference implementation to be checked against.
"""

import hashlib
import math
from collections import defaultdict


def _hash(user, item, salt=""):
    """A stable pseudo-random ordering. Not security; just decorrelation.

    `salt` gives a different-but-still-deterministic split, which is what lets
    the same data be scored several times over. One split of one dataset is a
    single draw, and a single draw of a recommendation score moves a long way:
    measured on noise it ranged from -36% to +86% lift.
    """
    h = hashlib.md5(f"{salt}\x00{user}\x00{item}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")

MIN_INTERACTIONS = 2      # a user needs this many to be splittable
DEFAULT_K = 5


# ------------------------------------------------------------------ shaping

def interactions(rows, user_col, item_col, weight_col=None):
    """Rows -> {user: {item: weight}}. Repeats are summed, not overwritten."""
    out = defaultdict(dict)
    for r in rows:
        u, i = r.get(user_col), r.get(item_col)
        if u is None or i is None or u == "" or i == "":
            continue
        w = 1.0
        if weight_col is not None:
            raw = r.get(weight_col)
            if raw is None or raw == "":
                continue
            try:
                w = float(raw)
            except (TypeError, ValueError):
                continue
        out[str(u)][str(i)] = out[str(u)].get(str(i), 0.0) + w
    return dict(out)


def split(data, holdout=0.3, min_train=1, salt=""):
    """Leave-k-out per user, deterministic and uncorrelated with item names.

    Which items are held back is chosen by a stable hash of (user, item), not
    by their sort order. Holding back the alphabetically-last items looked
    tidier and was quietly wrong: on uniformly random data it put the
    high-numbered items in the test set while the popularity baseline — which
    breaks ties alphabetically — recommended the low-numbered ones. The
    baseline was being handicapped by the split, and the model showed a 38%
    lift over pure noise. A hash decorrelates the two while staying
    reproducible, since no RNG state is involved.

    `skipped` carries the users too small to split — reporting them is the
    difference between an honest score and one padded with zeros.
    """
    train, test, skipped = {}, {}, []
    for u, items in data.items():
        if len(items) < MIN_INTERACTIONS:
            skipped.append(u)
            train[u] = dict(items)
            continue
        keys = sorted(items)
        n_hold = max(1, int(round(len(keys) * holdout)))
        n_hold = min(n_hold, len(keys) - min_train)
        if n_hold < 1:
            skipped.append(u)
            train[u] = dict(items)
            continue
        held = set(sorted(keys, key=lambda i: _hash(u, i, salt))[-n_hold:])
        train[u] = {k: items[k] for k in keys if k not in held}
        test[u] = {k: items[k] for k in held}
    return train, test, skipped


# -------------------------------------------------------------------- model

def item_similarity(train):
    """Cosine similarity between items, over the users who engaged with both."""
    by_item = defaultdict(dict)
    for u, items in train.items():
        for i, w in items.items():
            by_item[i][u] = w

    norms = {i: math.sqrt(sum(w * w for w in us.values()))
             for i, us in by_item.items()}

    # Only item pairs that share a user can be similar, so walk each user's
    # basket instead of every pair of items.
    dot = defaultdict(float)
    for u, items in train.items():
        names = list(items)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                i, jj = names[a], names[b]
                key = (i, jj) if i < jj else (jj, i)
                dot[key] += items[i] * items[jj]

    sim = defaultdict(dict)
    for (i, jj), d in dot.items():
        n = norms.get(i, 0.0) * norms.get(jj, 0.0)
        if n:
            s = d / n
            sim[i][jj] = s
            sim[jj][i] = s
    return dict(sim), by_item


def popularity(train):
    """How much engagement each item has overall — the baseline's whole model."""
    pop = defaultdict(float)
    for items in train.values():
        for i, w in items.items():
            pop[i] += w
    return dict(pop)


def recommend(user_items, sim, pop, k=DEFAULT_K, exclude_seen=True):
    """Top-k for one user: similarity to what they already have, else popular."""
    scores = defaultdict(float)
    for i, w in user_items.items():
        for jj, s in sim.get(i, {}).items():
            if exclude_seen and jj in user_items:
                continue
            scores[jj] += s * w
    if not scores:                       # cold user, or nothing similar
        scores = {i: p for i, p in pop.items()
                  if not (exclude_seen and i in user_items)}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(i, round(s, 6)) for i, s in ranked[:k]]


def recommend_popular(user_items, pop, k=DEFAULT_K, exclude_seen=True):
    ranked = sorted(((i, p) for i, p in pop.items()
                     if not (exclude_seen and i in user_items)),
                    key=lambda kv: (-kv[1], kv[0]))
    return [(i, round(p, 6)) for i, p in ranked[:k]]


# ----------------------------------------------------------------- accuracy

def _precision_recall(hits, k, n_relevant):
    p = hits / k if k else 0.0
    r = hits / n_relevant if n_relevant else 0.0
    return p, r


def _ap(ranked_ids, relevant):
    """Average precision: rewards putting the right items near the top."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for rank, i in enumerate(ranked_ids, start=1):
        if i in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), len(ranked_ids)) if hits else 0.0


def evaluate(data, k=DEFAULT_K, holdout=0.3, folds=1):
    """Fit on part of each user's history, score on what was held back.

    `folds` repeats the whole thing with a different split each time and
    averages — the recommendation counterpart of rolling-origin backtesting.
    The spread is reported too, because a lift of 38% means something quite
    different when the folds range 36-40 than when they range 10-70.
    """
    folds = max(1, int(folds))
    if folds > 1:
        runs = [_evaluate_once(data, k, holdout, salt=str(f))
                for f in range(folds)]
        runs = [r for r in runs if r]
        if not runs:
            return None
        out = dict(runs[0])
        num = [key for key, v in runs[0].items() if isinstance(v, (int, float))]
        for key in num:
            vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
            out[key] = round(sum(vals) / len(vals), 4) if vals else None
        lifts = [r["lift_over_popularity"] for r in runs
                 if isinstance(r.get("lift_over_popularity"), (int, float))]
        out.update({"folds": len(runs), "method": f"{len(runs)}-fold leave-k-out",
                    "k": k, "holdout": holdout})
        if lifts:
            out["lift_min"] = round(min(lifts), 2)
            out["lift_max"] = round(max(lifts), 2)
        return out
    return _evaluate_once(data, k, holdout)


def _evaluate_once(data, k=DEFAULT_K, holdout=0.3, salt=""):
    train, test, skipped = split(data, holdout=holdout, salt=salt)
    if not test:
        return None
    sim, _ = item_similarity(train)
    pop = popularity(train)

    m = {"precision": 0.0, "recall": 0.0, "map": 0.0}
    b = {"precision": 0.0, "recall": 0.0, "map": 0.0}
    covered, scored = set(), 0

    for u, held in test.items():
        relevant = set(held)
        recs = recommend(train.get(u, {}), sim, pop, k)
        base = recommend_popular(train.get(u, {}), pop, k)
        rec_ids = [i for i, _ in recs]
        base_ids = [i for i, _ in base]
        covered.update(rec_ids)

        p, r = _precision_recall(len(set(rec_ids) & relevant), k, len(relevant))
        m["precision"] += p; m["recall"] += r; m["map"] += _ap(rec_ids, relevant)
        p, r = _precision_recall(len(set(base_ids) & relevant), k, len(relevant))
        b["precision"] += p; b["recall"] += r; b["map"] += _ap(base_ids, relevant)
        scored += 1

    for d in (m, b):
        for key in d:
            d[key] = round(d[key] / scored, 4) if scored else 0.0

    lift = None
    if b["precision"]:
        lift = round((m["precision"] / b["precision"] - 1) * 100, 2)
    elif m["precision"]:
        lift = 100.0

    return {
        "method": "leave-k-out per user",
        "folds": 1,
        "k": k, "holdout": holdout,
        "users_scored": scored,
        "users_skipped": len(skipped),
        "items_known": len(pop),
        "catalog_coverage": round(len(covered) / len(pop), 4) if pop else 0.0,
        "precision_at_k": m["precision"],
        "recall_at_k": m["recall"],
        "map_at_k": m["map"],
        "popular_precision_at_k": b["precision"],
        "popular_recall_at_k": b["recall"],
        "popular_map_at_k": b["map"],
        "lift_over_popularity": lift,
    }


def verdict(acc):
    """trusted / weak / unpredictable — the same language forecasting uses."""
    if not acc or not acc.get("users_scored"):
        return ("not scored — no user had enough history to hold any "
                "interactions back")
    lift, p = acc.get("lift_over_popularity"), acc.get("precision_at_k") or 0
    if p == 0:
        return ("unpredictable — nothing recommended was actually engaged with; "
                "at this size that is as likely to be too little data as a bad model")
    if lift is None:
        return "scored, but popularity recommended nothing — no baseline to beat"
    if lift > 20:
        return "trusted — clearly beats recommending whatever is popular"
    if lift > 0:
        return "weak — beats popularity, but not by much"
    if lift == 0:
        return "tied with popularity — the personalisation is adding nothing"
    return "unpredictable — does worse than simply recommending popular items"


def fit_and_recommend(data, k=DEFAULT_K):
    """Fit on everything and produce the deliverable: top-k per user."""
    sim, _ = item_similarity(data)
    pop = popularity(data)
    out = {}
    for u, items in data.items():
        out[u] = recommend(items, sim, pop, k)
    return out, sim, pop
