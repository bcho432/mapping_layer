"""task_library: what a profile can ask of its data.

The profile core says what the data MEANS. The task says what is being ASKED of
it -- the clock, the keys, which metric is predicted, which ride alongside. Both
the profile-generator (which writes tasks) and the spec-compiler (which lowers
them) validate against this one table, so they cannot drift apart.

The field vocabulary is fixed and small:

    clock       {event, grain} -- the time axis, when there is one
    keys        the dimensions whose combination identifies one output row
    target      the metric predicted, for a numeric forward-looking question
    signal      the metric scored, when there is no ground-truth label
    label       the metric predicted, for a categorical question
    covariates  [{metric, availability}] -- carried alongside, not predicted

A kind declares only which of those it REQUIRES and which it PERMITS. Anything
else it has no concept of, so sending one is an error rather than a silently
ignored hint -- an optional field would mean every engine had to know to ignore
it, and nothing would catch a recommender that set a grain by mistake.

Adding an engine is a row here, not a module.

Same shape as function_library / derive_library: a sibling .csv when shipped,
an identical embedded copy otherwise. Stdlib only.
"""

import csv
import io
import os

CATALOG_FILENAME = "task_library.csv"

# Every field a task may carry, in a fixed order.
TASK_FIELDS = ("clock", "keys", "target", "signal", "label", "covariates")

# The three fields that name what is being predicted. They differ only in what
# the engine calls it; at most one is in any kind's `requires`.
POINTERS = ("target", "signal", "label")

# Availability of a covariate at prediction time. Profile-side only: it becomes
# an anchor in the SPEC and a hint for the run config.
AVAIL = ("known_ahead", "past_only")

# Keep identical to task_library.csv - the self_tests assert they match.
_EMBEDDED_CATALOG = """kind,requires,permits,keys_min,keys_max,description
forecast,clock|target,keys|covariates,0,4,Predict a metric forward in time
detect_anomaly,signal,clock|keys|covariates,0,4,Flag unusual values of a signal over time
classify,label|keys,clock|covariates,1,4,Predict a categorical label per entity
regress,target|keys,clock|covariates,1,4,Predict a numeric value per entity
cluster,keys,clock|covariates,1,4,Group entities by their covariates; nothing is predicted
recommend,keys|signal,covariates,2,2,Score the interaction between two entity keys
rank,keys|signal,covariates,2,2,Order items within a query key
"""

_catalog_cache = None


def _parse_catalog(text):
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        kind = (r.get("kind") or "").strip()
        if not kind:
            continue
        split = lambda s: tuple(x.strip() for x in (s or "").split("|") if x.strip())  # noqa: E731
        out[kind] = {
            "requires": split(r.get("requires")),
            "permits": split(r.get("permits")),
            "keys_min": int(r.get("keys_min") or 0),
            "keys_max": int(r.get("keys_max") or 99),
            "description": (r.get("description") or "").strip(),
        }
    return out


def embedded_catalog():
    """The catalog baked into this module (source of truth for the .csv)."""
    return _parse_catalog(_EMBEDDED_CATALOG)


def catalog():
    """The active catalog: the sibling .csv if present, else the embedded copy."""
    global _catalog_cache
    if _catalog_cache is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CATALOG_FILENAME)
        try:
            with open(path, encoding="utf-8") as f:
                _catalog_cache = _parse_catalog(f.read())
        except OSError:
            _catalog_cache = embedded_catalog()
    return _catalog_cache


def kinds():
    return tuple(catalog())


def plan(kind, what="task.kind"):
    """Look a kind up, or say what the known ones are. Never guesses."""
    lib = catalog()
    entry = lib.get(str(kind or "").strip().lower())
    if entry is None:
        raise ValueError(
            f"{what} must be one of {', '.join(sorted(lib))}, got '{kind}'")
    return entry


def allowed(p):
    return set(p["requires"]) | set(p["permits"])


def pointer_of(p):
    """Which field this kind uses to name what is predicted, or None.

    None is a real answer: clustering predicts nothing.
    """
    return next((f for f in POINTERS if f in p["requires"]), None)


def as_list(v):
    """A pointer may name one metric or several; normalise to a list."""
    if v is None:
        return []
    seq = v if isinstance(v, (list, tuple)) else [v]
    return [str(x).strip() for x in seq if str(x or "").strip()]


def check(task, metric_names, dim_names=(), what="task"):
    """Validate a task against its kind's row.

    Returns (plan, kind, pointer_field, pointed_metrics). Raises with a message
    that names the offending field and what the kind actually accepts.
    """
    if not isinstance(task, dict):
        raise ValueError(f"{what} must be an object")
    kind = str(task.get("kind") or "").strip().lower()
    p = plan(kind, f"{what}.kind")
    ok = allowed(p)

    for f in TASK_FIELDS:
        present = task.get(f) not in (None, "", [], {})
        if f in p["requires"] and not present:
            raise ValueError(f"{what} kind '{kind}' requires '{f}', but it is missing")
        if present and f not in ok:
            raise ValueError(
                f"{what} kind '{kind}' has no concept of '{f}' "
                f"(it accepts: {', '.join(sorted(ok))})")
    for k in task:
        if k != "kind" and k not in TASK_FIELDS:
            raise ValueError(
                f"{what} field '{k}' is not part of the task vocabulary "
                f"({', '.join(TASK_FIELDS)})")

    keys = task.get("keys") or []
    if not isinstance(keys, list):
        raise ValueError(f"{what}.keys must be a list of dimension names")
    if not p["keys_min"] <= len(keys) <= p["keys_max"]:
        raise ValueError(
            f"{what} kind '{kind}' needs {p['keys_min']}..{p['keys_max']} key(s), "
            f"got {len(keys)} ({keys})")
    for k in keys:
        if dim_names and str(k) not in dim_names:
            raise ValueError(
                f"{what}.keys names '{k}', which is not a declared dimension "
                f"({', '.join(sorted(dim_names))})")

    pointer = pointer_of(p)
    pointed = as_list(task.get(pointer)) if pointer else []
    if len(set(pointed)) != len(pointed):
        raise ValueError(f"{what}.{pointer} names the same metric twice")
    for pm in pointed:
        if metric_names and pm not in metric_names:
            raise ValueError(
                f"{what}.{pointer} names metric '{pm}', which is not declared "
                f"(metrics: {', '.join(sorted(metric_names))})")

    seen = set()
    for c in (task.get("covariates") or []):
        if not isinstance(c, dict):
            raise ValueError(f"{what}.covariates[] entries must be objects")
        cm = str(c.get("metric") or "").strip()
        if not cm:
            raise ValueError(f"{what}.covariates[] entries need a 'metric'")
        if metric_names and cm not in metric_names:
            raise ValueError(
                f"{what}.covariates names metric '{cm}', which is not declared")
        if cm in pointed:
            raise ValueError(
                f"{what}: metric '{cm}' is both predicted and a covariate")
        if cm in seen:
            raise ValueError(f"{what}.covariates names '{cm}' twice")
        seen.add(cm)
        av = str(c.get("availability") or "").strip().lower()
        if av and av not in AVAIL:
            raise ValueError(
                f"covariate '{cm}' availability must be one of {AVAIL}, got '{av}'")

    clock = task.get("clock")
    if clock is not None and not isinstance(clock, dict):
        raise ValueError(f"{what}.clock must be an object with event and grain")
    return p, kind, pointer, pointed
