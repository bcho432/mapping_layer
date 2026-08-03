"""The LLM stage: read the goal properly, correct the deterministic draft.

WHERE THIS SITS, AND WHY IT IS NOT INSIDE A SERVICE
---------------------------------------------------
    schema + goal
      -> profile-generator (suggest)     deterministic scaffold, stdlib only
      -> [ this module ]                 a model corrects the draft
      -> human edits + signs             the frontend
      -> profile-generator (finalize)    deterministic validate, stdlib only

The three services ship through git-publisher with no package_dependencies, and
every one of them says so in its own docstring. An HTTP call to Anthropic breaks
that, and would put an API key inside SPL at runtime. So the model goes in the
bench, in the slot the README already describes, with the generator sitting on
BOTH sides of it — the draft it corrects is deterministic, and the profile it
returns must survive `finalize`, which rejects unknown functions, unbound
columns and bad grains regardless of who wrote them.

WHAT REPLACES WHAT
------------------
The scaffold picks `kind`, `grain`, the function and the columns from keyword
regexes over the goal: `_RECOMMENDISH.search(...)`, `g in goal_l`,
`_fn_from_goal(...)`. That cannot read negation, unlisted synonyms, or intent —
"do NOT show me a monthly view" scored `month`, and "market fragmentation by
vendor" missed `hhi` because the word was absent from the table. A model reads
the sentence. It is strictly better at the reading and no better at the
validating, which is why the validating stays where it was.

NEVER FAILS OPEN
----------------
No key, no network, a refusal, malformed JSON, a hallucinated column, a function
outside the catalog — every one of those returns the deterministic draft
unchanged, with the reason recorded. The bench works with no API key at all;
the model makes it better, never load-bearing.

Requires the `anthropic` SDK. This module is the only thing in the tree that
does, and nothing under services/ imports it.
"""

import json
import os

MODEL = "claude-opus-5"
MAX_TOKENS = 8000

# A column is described to the model by what is actually in it — the same facts
# the scaffold reads, plus the sample values it ignores. Names are a hint;
# `Month` holding "1949-01" is a date whatever it is called.
SYSTEM = """\
You correct a draft analytics profile so it matches what the user asked for.

A profile says how to fold a table of rows into a frame: which columns identify
a row group (keys), what to measure in each group (metrics), and at what time
grain, if there is a clock at all. It does NOT say what to predict — that is a
run-config decision made later.

You are given the real column schema, the user's goal sentence, and a draft
produced by keyword matching. The draft is often right. Change only what the
goal actually calls for, and explain each change in one short clause.

Rules you cannot break:
- Every column you name must exist in the schema, spelled exactly.
- Every metric function must come from the catalog you are given.
- A function's `needs` must be satisfied: `measure` means set `measure` to a
  numeric column; `entity` means the metric also needs `entity` set to the
  column whose shares are being measured.
- kind `forecast` and `anomaly` require a time column; `recommend`, `cluster`,
  `classify`, `regress` and `rank` must not have one.
- `recommend` needs exactly two key columns: who, and what they engaged with.
- Every kind except forecast needs at least one key. When the file has no id
  column — each row is already the thing being described, as in a table of
  measurements — return the key name "__row_number__" instead of a column.
  Never return an empty key list: a frame with no key has no rows.
- Read the sentence, including negation. "not monthly" is not a vote for month.
- Prefer `mean` over `sum` for an intensive quantity — a temperature, a rate, a
  ratio, a score, an index. Summing 31 daily temperatures is meaningless.
- A numeric column that is really a code or an id (a NAICS code, a store
  number, a zip) is a key or an entity, never something to average.
- If the goal is vague, keep the draft. Do not invent an intent it lacks.\
"""

# Absent is spelled "" or [], never null: structured outputs support anyOf but a
# nullable union buys nothing here and costs schema surface.
SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string",
                 "enum": ["forecast", "anomaly", "classify", "cluster",
                          "regress", "recommend", "rank"]},
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "function": {"type": "string"},
                    "measure": {"type": "string"},
                    "entity": {"type": "string"},
                    "role": {"type": "string", "enum": ["target", "feature"]},
                },
                "required": ["name", "function", "measure", "entity", "role"],
                "additionalProperties": False,
            },
        },
        "time_event": {"type": "string",
                       "description": "the date column, or '' when there is no clock"},
        "time_grain": {"type": "string",
                       "enum": ["day", "week", "month", "quarter", "year", ""]},
        "series": {"type": "string",
                   "description": "column that names one series, or '' for a single series"},
        "keys": {"type": "array", "items": {"type": "string"},
                 "description": ("explicit key columns for clockless kinds; use "
                                 "'__row_number__' when each row is already the "
                                 "thing being described and no id column exists")},
        "dimensions": {"type": "array", "items": {"type": "string"}},
        "changes": {"type": "array", "items": {"type": "string"},
                    "description": "one short clause per change, empty if none"},
    },
    "required": ["kind", "metrics", "time_event", "time_grain", "series",
                 "keys", "dimensions", "changes"],
    "additionalProperties": False,
}


# Every place the SDK looks for a credential, in its own precedence order. An
# earlier version of this gate checked only ANTHROPIC_API_KEY, which quietly
# refused to run under the two credential sources that avoid a static key
# altogether — `ant auth login`'s OAuth profile and ANTHROPIC_AUTH_TOKEN.
# Checking one source and calling it "no credentials" is how a working setup
# gets reported as a missing one.
_CONFIG_DIR = os.environ.get("ANTHROPIC_CONFIG_DIR") or os.path.expanduser(
    "~/.config/anthropic")


def credential_source(api_key=None):
    """Which credential the SDK will use, without ever reading its value."""
    if api_key:
        return "an explicitly passed key"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY"
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "ANTHROPIC_AUTH_TOKEN"
    if os.path.isdir(os.path.join(_CONFIG_DIR, "credentials")):
        prof = os.environ.get("ANTHROPIC_PROFILE") or "default"
        if os.path.exists(os.path.join(_CONFIG_DIR, "credentials",
                                       f"{prof}.json")):
            return f"the '{prof}' OAuth profile from `ant auth login`"
    return ""


def available(api_key=None):
    """Is the model reachable at all? Decided once, reported to the UI."""
    src = credential_source(api_key)
    if not src:
        return False, ("no credentials — set ANTHROPIC_API_KEY, or run "
                       "`ant auth login` for a keyless OAuth profile")
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "the anthropic SDK is not installed (pip install anthropic)"
    return True, f"ready via {src}"


def enrich(draft, schema, goal, catalog, binding_stub=None, api_key=None,
           model=MODEL):
    """Correct `draft` against `goal`. Returns (profile, report).

    Falls back to the draft unchanged on every failure path. The report always
    says which happened and why, so a silent degradation is impossible to
    mistake for a successful call.
    """
    report = {"used": False, "model": model, "changes": [], "reason": "",
              "credential": credential_source(api_key) or "none"}

    ok, why = available(api_key)
    if not ok:
        report["reason"] = why
        return draft, report
    if not str(goal or "").strip():
        report["reason"] = "no goal to read — nothing for a model to improve on"
        return draft, report

    try:
        return _enrich(draft, schema, goal, catalog, api_key, model, report)
    except Exception as e:                    # noqa: BLE001 — deliberate
        # The bench halts the pipeline on ANY unhandled stage exception, so a
        # named-exception list is not a safety net here: one unnamed error
        # class and a degradation becomes an outage. This module's whole
        # contract is that it cannot break the run, so it catches everything —
        # including anything raised while BUILDING the request, which an
        # inner try around the API call alone would have missed.
        report["reason"] = f"{type(e).__name__}: {e}"
        return draft, report


def _enrich(draft, schema, goal, catalog, api_key, model, report):
    import anthropic

    prompt = _prompt(draft, schema, goal, catalog)
    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key \
            else anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIStatusError as e:
        report["reason"] = f"API {e.status_code}: {e.message}"
        return draft, report
    except anthropic.APIConnectionError:
        report["reason"] = "could not reach the API — kept the scaffold"
        return draft, report


    # A refusal is a successful HTTP call with no content. Reading content[0]
    # unconditionally is how that turns into a crash instead of a fallback.
    if resp.stop_reason == "refusal":
        report["reason"] = "the model declined this request"
        return draft, report
    if resp.stop_reason == "max_tokens":
        report["reason"] = "the reply hit max_tokens and is incomplete"
        return draft, report

    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        got = json.loads(text)
    except json.JSONDecodeError as e:
        report["reason"] = f"the reply was not JSON: {e}"
        return draft, report

    profile, problems = _apply(draft, got, schema, catalog)
    if problems:
        # The model named something that does not exist. That is exactly what
        # the deterministic side is for — take the draft, say why.
        report["reason"] = "rejected: " + "; ".join(problems[:3])
        report["rejected"] = problems
        return draft, report

    report.update({"used": True, "changes": got.get("changes") or [],
                   "reason": "the model read the goal",
                   "usage": {"input": resp.usage.input_tokens,
                             "output": resp.usage.output_tokens}})
    return profile, report


# ------------------------------------------------------------------- prompt

def _prompt(draft, schema, goal, catalog):
    cols = []
    for c in schema:
        bits = [f"  {c['name']}", f"type={c.get('type', '?')}"]
        if c.get("sample") not in (None, ""):
            bits.append(f"sample={c['sample']!r}")
        if isinstance(c.get("cardinality"), (int, float)):
            bits.append(f"distinct_ratio={c['cardinality']:.2f}")
        cols.append("  ".join(bits))

    fns = [f"  {n:<12} needs={list(e['needs']) or 'nothing'}  {e['description']}"
           for n, e in sorted(catalog.items())]

    return (
        f"GOAL (the user's own words)\n{goal.strip()}\n\n"
        f"COLUMNS IN THE FILE\n" + "\n".join(cols) + "\n\n"
        f"FUNCTION CATALOG (the only functions that exist)\n" + "\n".join(fns) +
        "\n\nDRAFT PROFILE (from keyword matching — correct it)\n"
        + json.dumps(_summarise(draft), indent=2) +
        "\n\nReturn the corrected profile. If the draft already matches the "
        "goal, return it unchanged with an empty `changes` list."
    )


def _keep_draft_keys(draft, got, schema, catalog, names):
    """Re-apply the reply with the draft's keys restored."""
    patched = dict(got)
    patched["keys"] = []
    prof, probs = _apply(draft, patched, schema, catalog)
    if probs:
        return draft, probs
    prof["keys"] = list(draft.get("keys") or [])
    return prof, []


def _seq(v):
    """A list, whatever arrived. A string is a scalar here, never characters."""
    return v if isinstance(v, list) else []


def _summarise(draft):
    """The draft in the same vocabulary the reply uses, so the diff is legible."""
    draft = draft if isinstance(draft, dict) else {}
    t = draft.get("time") or {}
    xd = draft.get("x-deep") or draft.get("x_deep") or {}
    return {
        "kind": draft.get("kind") or "forecast",
        "metrics": [{k: m.get(k, "") for k in
                     ("name", "function", "measure", "entity", "role")}
                    for m in _seq(draft.get("metrics")) if isinstance(m, dict)],
        "time_event": t.get("event", ""),
        "time_grain": t.get("grain", ""),
        "series": xd.get("series", ""),
        "keys": [k.get("from", "") for k in _seq(draft.get("keys"))
                 if isinstance(k, dict)],
        "dimensions": [d.get("name", "") for d in _seq(draft.get("dimensions"))
                       if isinstance(d, dict)],
    }


# -------------------------------------------------------------- validation

def _apply(draft, got, schema, catalog):
    """Rebuild a profile from the reply, refusing anything that isn't real.

    Structured outputs make a wrong shape unlikely, not impossible — and this
    also runs against hand-edited replies in the tests. Every field is read
    through a type check rather than assumed.
    """
    if not isinstance(got, dict):
        return draft, [f"the reply was {type(got).__name__}, not an object"]
    names = {c["name"] for c in schema if isinstance(c, dict) and "name" in c}
    problems = []

    kind = str(got.get("kind") or "forecast").strip().lower()
    metrics = []
    for i, m in enumerate(_seq(got.get("metrics"))):
        if not isinstance(m, dict):
            problems.append(f"metrics[{i}] is not an object")
            continue
        fn = str(m.get("function") or "").strip()
        entry = catalog.get(fn)
        if not entry:
            problems.append(f"metrics[{i}].function {fn!r} is not in the catalog")
            continue
        needs = set(entry["needs"])
        out = {"name": str(m.get("name") or f"m{i}").strip(),
               "function": fn,
               "role": str(m.get("role") or "target").strip().lower()}
        for field in ("measure", "entity"):
            val = str(m.get(field) or "").strip()
            if field in needs:
                if val not in names:
                    problems.append(
                        f"metrics[{i}].{field} {val!r} is not a column"
                        if val else f"metrics[{i}] needs a {field}")
                else:
                    out[field] = val
            elif val:
                problems.append(f"metrics[{i}] sets {field} but {fn} does not use it")
        metrics.append(out)
    if not metrics:
        problems.append("no usable metric came back")
    elif not any(m["role"] == "target" for m in metrics):
        # finalize refuses a profile with nothing to predict. Catching it here
        # turns a hard stage failure into a fallback, which is the whole point.
        problems.append("no metric has role 'target'")

    event = str(got.get("time_event") or "").strip()
    grain = str(got.get("time_grain") or "").strip().lower()
    if event and event not in names:
        problems.append(f"time_event {event!r} is not a column")
    if event and not grain:
        problems.append("a time column was given with no grain")
    if kind in ("forecast", "anomaly") and not event:
        problems.append(f"kind {kind!r} needs a time column")
    if kind not in ("forecast", "anomaly") and event:
        problems.append(f"kind {kind!r} must not have a clock")

    ROW = "__row_number__"
    keys = [str(k) for k in _seq(got.get("keys")) if str(k).strip()]
    for k in keys:
        if k != ROW and k not in names:
            problems.append(f"keys names {k!r}, which is not a column")
    # An empty key list is never an intent — a frame with no key has no rows,
    # and the compiler refuses it. Silently inheriting the draft's keys is the
    # right recovery: the model was correcting the metrics, not deleting the
    # grain. This is the same class as a hallucinated column, caught earlier.
    if not keys and kind != "forecast":
        draft_keys = draft.get("keys") if isinstance(draft, dict) else None
        if isinstance(draft_keys, list) and draft_keys:
            return _keep_draft_keys(draft, got, schema, catalog, names)
        problems.append(f"kind {kind!r} needs at least one key; none were given")
    if kind == "recommend" and len(keys) != 2:
        problems.append(
            f"kind 'recommend' needs exactly two keys (who, what); got {len(keys)}")

    series = str(got.get("series") or "").strip()
    if series and series not in names:
        problems.append(f"series {series!r} is not a column")

    if problems:
        return draft, problems

    dims = [d for d in _seq(got.get("dimensions")) if d in names]
    for extra in ([series] if series else []) + keys:
        if extra and extra != ROW and extra not in dims:
            dims.append(extra)

    profile = {"name": draft.get("name") or "profile",
               "version": draft.get("version") or "1",
               "datasets": draft.get("datasets") or [{"name": "dataset"}],
               "metrics": metrics,
               "dimensions": [{"name": d} for d in dims]}
    if kind != "forecast":
        profile["kind"] = kind
    if event:
        profile["time"] = {"event": event, "grain": grain}
    if keys:
        profile["keys"] = [{"name": "k_row", "via": "row_number"} if k == ROW
                           else {"name": f"k_{k}", "from": k} for k in keys]
    if series:
        profile["x-deep"] = {"series": series}
    return profile, []


def rebind(binding_stub, profile, source):
    """Widen the binding so every column the model chose is actually bound."""
    bind = dict((binding_stub or {}).get("bind") or {})
    for m in _seq(profile.get("metrics")):
        for field in ("measure", "entity"):
            if m.get(field):
                bind.setdefault(m[field], m[field])
    for k in _seq(profile.get("keys")):
        if k.get("from"):
            bind.setdefault(k["from"], k["from"])
    t = profile.get("time") or {}
    if t.get("event"):
        bind.setdefault(t["event"], t["event"])
    xd = profile.get("x-deep") or {}
    if xd.get("series"):
        bind.setdefault(xd["series"], xd["series"])
    return {"source": (binding_stub or {}).get("source") or source,
            "bind": bind,
            "available_from": (binding_stub or {}).get("available_from", "")}
