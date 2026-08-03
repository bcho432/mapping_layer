"""Random forest classification, stdlib only.

The third engine, after ets.py (forecasting) and recommend.py (recommendation),
and it keeps the same discipline: a real model, a real baseline, and a score
measured on rows held back from fitting.

THE MODEL. A decision tree splits the data on one feature at a time, choosing
the split that most reduces Gini impurity — the chance that two rows drawn from
the same node have different labels. A forest grows many such trees, each on a
bootstrap sample of the rows and each considering only a random subset of the
features at every split. Those two sources of randomness are what stop the
trees agreeing with each other, and a vote among disagreeing trees generalises
better than any single tree that fitted the training set perfectly.

THE BASELINE. Predict the most common class, every time. This is the naive
forecast of classification: free, and on imbalanced data embarrassingly hard to
beat. 80% accuracy means nothing if 80% of the rows are one class, which is
exactly the trap that makes accuracy alone the wrong number to report.

THE SPLIT. Stratified k-fold: every fold holds the same class mix as the whole,
so a rare class cannot vanish from a fold and leave its recall undefined. Each
row is predicted exactly once, by a forest that never saw it.

WHY NOT SCIKIT-LEARN. The services ship through git-publisher with no
package_dependencies. That constraint is the reason this file exists, and it is
worth being clear that a real deployment should use scikit-learn: it is faster,
better tested, and has twenty years of edge cases baked in. This exists so the
pipeline can be judged end to end without one.
"""

import math
import random
from collections import Counter

DEFAULT_TREES = 60
DEFAULT_DEPTH = 8
MIN_SPLIT = 4          # a node with fewer rows than this becomes a leaf
MIN_PER_CLASS = 2      # a class with fewer rows than this cannot be stratified


# ------------------------------------------------------------------- the tree

def _gini(counts, n):
    """Impurity of a node: the chance two random rows disagree on the label."""
    if n <= 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _best_split(rows, labels, idx, feature_pool, rng):
    """The (feature, threshold) that most reduces impurity, or None.

    Thresholds are midpoints between adjacent distinct values, so a split can
    never sit exactly on an observed value — that keeps `<=` unambiguous when
    the same number appears on both sides.
    """
    n = len(idx)
    parent = Counter(labels[i] for i in idx)
    base = _gini(parent, n)
    best = None
    best_gain = 1e-12                       # a split must actually improve

    k = max(1, int(round(math.sqrt(len(feature_pool)))))
    for f in rng.sample(feature_pool, min(k, len(feature_pool))):
        vals = sorted({rows[i][f] for i in idx})
        if len(vals) < 2:
            continue
        for a, b in zip(vals, vals[1:]):
            thr = (a + b) / 2.0
            left = [i for i in idx if rows[i][f] <= thr]
            if not left or len(left) == n:
                continue
            right = [i for i in idx if rows[i][f] > thr]
            lc = Counter(labels[i] for i in left)
            rc = Counter(labels[i] for i in right)
            after = (len(left) / n) * _gini(lc, len(left)) + \
                    (len(right) / n) * _gini(rc, len(right))
            gain = base - after
            if gain > best_gain:
                best_gain, best = gain, (f, thr, left, right)
    return best


def _grow(rows, labels, idx, feature_pool, depth, rng, weights):
    """Grow one node. A leaf carries the weighted class distribution."""
    counts = Counter(labels[i] for i in idx)
    if depth <= 0 or len(idx) < MIN_SPLIT or len(counts) == 1:
        return {"leaf": _distribution(counts, weights)}
    split = _best_split(rows, labels, idx, feature_pool, rng)
    if split is None:
        return {"leaf": _distribution(counts, weights)}
    f, thr, left, right = split
    return {"f": f, "thr": thr,
            "l": _grow(rows, labels, left, feature_pool, depth - 1, rng, weights),
            "r": _grow(rows, labels, right, feature_pool, depth - 1, rng, weights)}


def _distribution(counts, weights):
    """Class probabilities in this leaf, after class weighting.

    `weights` is how much one row of each class counts for. Balanced weighting
    is what stops a forest on 8%-positive data from learning to say `no` every
    time and calling 92% accuracy a success.
    """
    tot = sum(counts[c] * weights.get(c, 1.0) for c in counts)
    if tot <= 0:
        return {c: 1.0 / len(counts) for c in counts}
    return {c: counts[c] * weights.get(c, 1.0) / tot for c in counts}


def _ask(node, row):
    while "leaf" not in node:
        node = node["l"] if row[node["f"]] <= node["thr"] else node["r"]
    return node["leaf"]


# ----------------------------------------------------------------- the forest

def fit(rows, labels, n_trees=DEFAULT_TREES, depth=DEFAULT_DEPTH,
        class_weight=None, seed=0):
    """Grow `n_trees` on bootstrap samples with random feature subsets."""
    rng = random.Random(seed)
    n, n_feat = len(rows), len(rows[0])
    pool = list(range(n_feat))

    weights = {}
    if class_weight == "balanced":
        counts = Counter(labels)
        k = len(counts)
        for c, cnt in counts.items():
            weights[c] = n / (k * cnt) if cnt else 1.0

    trees = []
    for t in range(n_trees):
        boot = [rng.randrange(n) for _ in range(n)]
        trees.append(_grow(rows, labels, boot, pool, depth,
                           random.Random(seed + t), weights))
    return {"trees": trees, "classes": sorted(set(labels)), "weights": weights}


def predict_proba(model, row):
    """Average the trees' leaf distributions — a vote weighted by confidence."""
    total = {c: 0.0 for c in model["classes"]}
    for tree in model["trees"]:
        for c, p in _ask(tree, row).items():
            total[c] = total.get(c, 0.0) + p
    n = len(model["trees"])
    return {c: v / n for c, v in total.items()}


def predict(model, row, positive=None, threshold=0.5):
    """The predicted class. Binary problems honour `threshold`.

    A threshold only means something when there are two classes and you have
    said which one you care about: it is the point where a false alarm becomes
    cheaper than a miss. With three or more classes the highest probability
    wins and the threshold is meaningless, so it is ignored rather than
    silently doing something arbitrary.
    """
    proba = predict_proba(model, row)
    if positive is not None and len(model["classes"]) == 2:
        other = next(c for c in model["classes"] if c != positive)
        return positive if proba.get(positive, 0.0) >= threshold else other
    return max(proba, key=lambda c: (proba[c], c))


# ----------------------------------------------------------- cross-validation

def stratified_folds(labels, k, seed=0):
    """Split indices into k folds, each holding the whole set's class mix.

    Without this a rare class can land entirely in one fold, and every other
    fold scores it as undefined rather than badly — which reads as a bug in the
    metrics rather than a shortage of data.
    """
    rng = random.Random(seed)
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    folds = [[] for _ in range(k)]
    for y in sorted(by_class):
        idx = by_class[y][:]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return [sorted(f) for f in folds if f]


def evaluate(rows, labels, k=5, n_trees=DEFAULT_TREES, depth=DEFAULT_DEPTH,
             class_weight=None, positive=None, threshold=0.5, seed=0):
    """Predict every row once, from a forest that never saw it."""
    counts = Counter(labels)
    if len(counts) < 2:
        return None, "every row has the same label — there is nothing to separate"
    too_small = {c for c, n in counts.items() if n < MIN_PER_CLASS}
    if too_small:
        return None, (f"class {sorted(too_small)[0]!r} has fewer than "
                      f"{MIN_PER_CLASS} rows — too few to hold any back")
    k = max(2, min(int(k), min(counts.values())))

    folds = stratified_folds(labels, k, seed)
    pred, proba = [None] * len(labels), [None] * len(labels)
    for f, test in enumerate(folds):
        train = [i for i in range(len(labels)) if i not in set(test)]
        model = fit([rows[i] for i in train], [labels[i] for i in train],
                    n_trees=n_trees, depth=depth, class_weight=class_weight,
                    seed=seed + f)
        for i in test:
            pred[i] = predict(model, rows[i], positive, threshold)
            proba[i] = predict_proba(model, rows[i])
    return {"pred": pred, "proba": proba, "folds": len(folds)}, ""


# ----------------------------------------------------------------- accuracy

def score(labels, pred, proba, positive=None):
    """Per-class precision/recall/F1, a confusion matrix, and the baseline."""
    classes = sorted(set(labels) | {p for p in pred if p is not None})
    n = len(labels)
    correct = sum(1 for y, p in zip(labels, pred) if y == p)

    # The baseline: always answer with the most common class.
    counts = Counter(labels)
    majority = max(counts, key=lambda c: (counts[c], c))
    base = counts[majority] / n if n else 0.0

    confusion = {a: {b: 0 for b in classes} for a in classes}
    for y, p in zip(labels, pred):
        if p is not None:
            confusion[y][p] += 1

    per_class, f1s = {}, []
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[a][c] for a in classes if a != c)
        fn = sum(confusion[c][b] for b in classes if b != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[str(c)] = {"precision": round(prec, 4), "recall": round(rec, 4),
                             "f1": round(f1, 4), "support": counts.get(c, 0)}
        f1s.append(f1)

    acc = correct / n if n else 0.0
    out = {"accuracy": round(acc, 4),
           "majority_baseline": round(base, 4),
           "majority_class": str(majority),
           "lift_over_baseline": round((acc / base - 1) * 100, 2) if base else None,
           "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
           "per_class": per_class,
           "confusion": {str(a): {str(b): confusion[a][b] for b in classes}
                         for a in classes},
           "rows_scored": n, "n_classes": len(classes)}
    if positive is not None and len(classes) == 2:
        out["roc_auc"] = _auc(labels, proba, positive)
        out["positive_class"] = str(positive)
    return out


def _auc(labels, proba, positive):
    """Rank-based AUC: the chance a positive outranks a negative.

    Computed from ranks rather than by sweeping thresholds, so it needs no
    threshold of its own — which is the point of reporting it next to a metric
    that does.
    """
    scored = [(p.get(positive, 0.0), 1 if y == positive else 0)
              for y, p in zip(labels, proba) if p is not None]
    pos = sum(t for _, t in scored)
    neg = len(scored) - pos
    if not pos or not neg:
        return None
    scored.sort(key=lambda t: t[0])
    rank, i = {}, 0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0             # average rank over ties
        for m in range(i, j + 1):
            rank[m] = avg
        i = j + 1
    rsum = sum(rank[m] for m, (_, t) in enumerate(scored) if t)
    return round((rsum - pos * (pos + 1) / 2) / (pos * neg), 4)


def verdict(acc):
    """trusted / weak / unpredictable — the language the other engines use."""
    if not acc:
        return "not scored"
    lift = acc.get("lift_over_baseline")
    macro = acc.get("macro_f1", 0.0)
    if lift is None:
        return "scored, but every row is one class — no baseline to beat"
    if lift <= 0:
        return ("unpredictable — does no better than always answering "
                f"{acc['majority_class']!r}")
    if lift > 10 and macro > 0.6:
        return "trusted — beats predicting the majority class by a clear margin"
    if lift > 10:
        return ("mixed — beats the baseline overall, but at least one class is "
                "predicted badly (see macro F1)")
    return "weak — beats the majority class, but not by much"
