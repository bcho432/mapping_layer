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
You choose which question a user is asking of their data, and correct a draft
analytics profile so it matches.

A profile splits in two. The CORE says what the data MEANS: what to measure in
each row group (metrics), and which columns identify one (keys). The TASK says
what is being ASKED of that core: which engine, whether there is a clock, and
which metric is the thing being predicted.

Choosing the engine is the most important thing you do, because everything else
follows from it — whether a clock exists at all, how many key columns to pick,
and whether anything is predicted. Choose it from what the user wants to KNOW,
not from which words happen to appear. "Which customers are about to leave" is
a classification whether or not it says the word.

You are given the real column schema, the user's goal, the engines that exist,
the functions that exist, and a draft produced by keyword matching. The draft
is a floor, not a reading: it cannot tell that "at risk of withdrawing" means
the WITHDRAWN_FLAG column, and you can. Change what the goal calls for, and
explain each change in one short clause.

Rules you cannot break:
- Every column you name must exist in the schema, spelled exactly.
- Every metric function must come from the catalog you are given.
- A function's `needs` must be satisfied: `measure` means set `measure` to a
  numeric column; `entity` means the metric also needs `entity` set to the
  column whose shares are being measured.
- Obey the engine table. A field the engine REQUIRES must be present; one it
  does not accept must be absent; the key count must fall in its range. An
  engine that accepts a clock does not need one — give it a clock only if the
  goal asks about time.
- Mark what is predicted with role `target`, and everything carried alongside
  with role `feature`. An engine that predicts nothing takes no target at all.
- When the file has no id column — each row is already the thing being
  described, as in a table of measurements — return the key name
  "__row_number__" instead of a column. Never return an empty key list: a frame
  with no key has no rows.
- Read the sentence, including negation. "not monthly" is not a vote for month.
- Prefer `mean` over `sum` for an intensive quantity — a temperature, a rate, a
  ratio, a score, an index. Summing 31 daily temperatures is meaningless.
- A numeric column that is really a code or an id (a NAICS code, a store
  number, a zip) is a key or an entity, never something to average.
- If the goal is vague, keep the draft. Do not invent an intent it lacks.\
"""

def engine_table(tasks):
    """The engines, as the model is shown them.

    Generated from `task_library`, never written out by hand. The prompt used to
    carry the rules as prose — "forecast and anomaly require a time column;
    recommend, cluster, classify, regress and rank must not have one" — which
    was already wrong against the catalog in three places by the time it was
    read. A generated table cannot drift: adding an engine is a row, and the
    row reaches the model in the same commit it reaches the compiler.
    """
    out = []
    for kind, e in tasks.items():
        keys = (str(e["keys_min"]) if e["keys_min"] == e["keys_max"]
                else f"{e['keys_min']}-{e['keys_max']}")
        out.append(
            f"  {kind}\n"
            f"      {e['description']}\n"
            f"      requires: {', '.join(e['requires']) or 'nothing'}\n"
            f"      accepts:  {', '.join(e['permits']) or 'nothing beyond what it requires'}\n"
            f"      keys:     {keys}")
    return "\n".join(out)


# What each field in that table means. The model is told the vocabulary once,
# rather than having to infer it from seven rows of it.
FIELD_GLOSSARY = """\
  clock       a time axis: {event column, grain}. Only for an engine that
              requires or accepts one, and only when the goal is about time.
  keys        the columns whose combination identifies one output row.
  target      the metric being predicted. Set role 'target' on it.
  signal      the metric being scored, where there is no ground truth. Same
              thing: set role 'target' on it and the pipeline renames it.
  label       the metric being predicted, when it is a category. Same again —
              role 'target'. You never have to know which of the three a given
              engine calls it.
  covariates  metrics carried alongside but never predicted. Role 'feature'.\
"""


def reply_schema(tasks):
    """The structured-output schema, with the kind enum read off the catalog."""
    s = json.loads(json.dumps(_REPLY_SCHEMA))
    s["properties"]["kind"]["enum"] = list(tasks)
    return s


# Absent is spelled "" or [], never null: structured outputs support anyOf but a
# nullable union buys nothing here and costs schema surface.
_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": []},
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


def enrich(draft, schema, goal, catalog, tasks, binding_stub=None, api_key=None,
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
        return _enrich(draft, schema, goal, catalog, tasks, api_key, model, report)
    except Exception as e:                    # noqa: BLE001 — deliberate
        # The bench halts the pipeline on ANY unhandled stage exception, so a
        # named-exception list is not a safety net here: one unnamed error
        # class and a degradation becomes an outage. This module's whole
        # contract is that it cannot break the run, so it catches everything —
        # including anything raised while BUILDING the request, which an
        # inner try around the API call alone would have missed.
        report["reason"] = f"{type(e).__name__}: {e}"
        return draft, report


def _enrich(draft, schema, goal, catalog, tasks, api_key, model, report):
    import anthropic

    prompt = _prompt(draft, schema, goal, catalog, tasks)
    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key \
            else anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema",
                                      "schema": reply_schema(tasks)}},
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

    profile, problems = _apply(draft, got, schema, catalog, tasks)
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

def _prompt(draft, schema, goal, catalog, tasks):
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
        f"ENGINES (the only questions that can be asked; pick exactly one)\n"
        + engine_table(tasks) + "\n\n"
        f"WHAT THOSE FIELDS MEAN\n" + FIELD_GLOSSARY + "\n\n"
        f"COLUMNS IN THE FILE\n" + "\n".join(cols) + "\n\n"
        f"FUNCTION CATALOG (the only functions that exist)\n" + "\n".join(fns) +
        "\n\nDRAFT PROFILE (from keyword matching — correct it)\n"
        + json.dumps(_summarise(draft), indent=2) +
        "\n\nReturn the corrected profile. If the draft already matches the "
        "goal, return it unchanged with an empty `changes` list."
    )


def _keep_draft_keys(draft, got, schema, catalog, tasks, draft_keys):
    """Re-apply the reply with the draft's keys restored.

    The keys come from `_summarise`, so they are column NAMES whichever shape
    the draft was written in — reading draft["keys"] directly put objects back
    into a field that now holds names.
    """
    patched = dict(got)
    patched["keys"] = list(draft_keys)
    prof, probs = _apply(draft, patched, schema, catalog, tasks)
    if probs:
        return draft, probs
    return prof, []


def _seq(v):
    """A list, whatever arrived. A string is a scalar here, never characters."""
    return v if isinstance(v, list) else []


def _summarise(draft):
    """The draft in the same vocabulary the reply uses, so the diff is legible.

    Reads either profile shape. A split profile keeps the question in `task`
    and carries no `role` on its metrics, so reading only the flat spelling
    showed the model an empty draft — no kind, no clock, no keys — and silently
    threw away everything the scaffold had worked out. The model then had to
    re-derive it from the goal alone, which is exactly the help it was supposed
    to be given.
    """
    draft = draft if isinstance(draft, dict) else {}
    task = draft.get("task") if isinstance(draft.get("task"), dict) else {}
    clock = task.get("clock") if isinstance(task.get("clock"), dict) else {}
    t = clock or (draft.get("time") or {})
    xd = draft.get("x-deep") or draft.get("x_deep") or {}

    # Which metric the task points at, whatever the engine calls the pointer.
    predicted = set()
    for f in ("target", "signal", "label"):
        v = task.get(f)
        predicted.update(v if isinstance(v, list) else [v] if v else [])
    carried = {c.get("metric") for c in _seq(task.get("covariates"))
               if isinstance(c, dict)}

    metrics = []
    for m in _seq(draft.get("metrics")):
        if not isinstance(m, dict):
            continue
        name = m.get("name", "")
        role = m.get("role", "")
        if not role and task:
            role = "target" if name in predicted else "feature" if name in carried else ""
        metrics.append({"name": name, "function": m.get("function", ""),
                        "measure": m.get("measure", ""),
                        # the core spells the share column `by`; the reply
                        # spells it `entity`, as the function catalog does
                        "entity": m.get("entity", m.get("by", "")),
                        "role": role})

    keys = []
    for k in _seq(task.get("keys") or draft.get("keys")):
        if isinstance(k, str):
            keys.append(k)
        elif isinstance(k, dict):
            via = str(k.get("via") or "")
            if via.startswith("bin:"):
                continue                      # that is the clock, not a key
            keys.append(k.get("from") or ("__row_number__" if via else ""))

    return {
        "kind": task.get("kind") or draft.get("kind") or "forecast",
        "metrics": metrics,
        "time_event": t.get("event", ""),
        "time_grain": t.get("grain", ""),
        "series": xd.get("series", ""),
        "keys": [k for k in keys if k],
        "dimensions": [d.get("name", "") for d in _seq(draft.get("dimensions"))
                       if isinstance(d, dict)],
    }


# -------------------------------------------------------------- validation

def _apply(draft, got, schema, catalog, tasks):
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

    # Every kind rule below is read off the catalog, not written out here. The
    # hardcoded pair this replaced ("forecast and anomaly need a clock; nothing
    # else may have one") disagreed with the catalog on three of seven engines
    # — it forbade a clock to classify, cluster and regress, which all accept
    # one — and it spelled detect_anomaly "anomaly".
    plan = tasks.get(kind)
    if plan is None:
        problems.append(
            f"kind {kind!r} is not an engine (have: {', '.join(sorted(tasks))})")
        return draft, problems
    accepts = set(plan["requires"]) | set(plan["permits"])

    event = str(got.get("time_event") or "").strip()
    grain = str(got.get("time_grain") or "").strip().lower()
    if event and event not in names:
        problems.append(f"time_event {event!r} is not a column")
    if event and not grain:
        problems.append("a time column was given with no grain")
    if event and "clock" not in accepts:
        problems.append(f"kind {kind!r} has no concept of a clock")
    if not event and "clock" in plan["requires"]:
        problems.append(f"kind {kind!r} requires a clock, but no time column was given")

    ROW = "__row_number__"
    keys = [str(k) for k in _seq(got.get("keys")) if str(k).strip()]
    for k in keys:
        if k != ROW and k not in names:
            problems.append(f"keys names {k!r}, which is not a column")
    # An empty key list is never an intent — a frame with no key has no rows,
    # and the compiler refuses it. Silently inheriting the draft's keys is the
    # right recovery: the model was correcting the metrics, not deleting the
    # grain. This is the same class as a hallucinated column, caught earlier.
    if not keys and plan["keys_min"] > 0:
        draft_keys = _summarise(draft)["keys"]
        if draft_keys:
            return _keep_draft_keys(draft, got, schema, catalog, tasks, draft_keys)
        problems.append(f"kind {kind!r} needs at least one key; none were given")
    elif not plan["keys_min"] <= len(keys) <= plan["keys_max"]:
        problems.append(
            f"kind {kind!r} takes {plan['keys_min']}..{plan['keys_max']} key(s); "
            f"got {len(keys)}")

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
        spelled = [{"name": "k_row", "via": "row_number"} if k == ROW
                   else {"name": f"k_{k}", "from": k} for k in keys]
        # An explicit keys[] is the general form: it carries every key the frame
        # has, the clock included. The sugar form derives the bin key from the
        # grain, but nothing derives it here, so a grained clock alongside
        # explicit keys produced a SPEC with a time column and nothing to bucket
        # it by. Only reachable since a classification was allowed a clock.
        if grain:
            spelled.append({"name": "t", "via": "bin:" + grain})
        profile["keys"] = spelled
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


# ------------------------------------------------------------------ self-test

def self_test(tasks, catalog, finalize=None, compile_fn=None):
    """Offline check: everything except the API call.

    The network half cannot be tested without a credential, and nothing
    interesting lives there — the reading is the model's job, the judging is
    this module's. What is checked here is the judging: that the prompt
    describes the engines that actually exist, that a draft is summarised
    without losing what the scaffold decided, and that every rule for accepting
    a reply comes from the catalog rather than a tuple written out by hand.

    `finalize` and `compile_fn`, when supplied, carry an accepted reply the rest
    of the way down. That is the check that matters most: a reply this module
    approves must survive the two deterministic stages after it.
    """
    checks = {}

    COLS = [
        {"name": "order_dt", "type": "date"},
        {"name": "cust_id", "type": "string", "cardinality": 500},
        {"name": "item_sku", "type": "string", "cardinality": 120},
        {"name": "stars", "type": "number"},
        {"name": "status", "type": "string", "cardinality": 3},
    ]

    # ---- the prompt describes the engines that exist -----------------------
    table = engine_table(tasks)
    checks["every catalogued engine reaches the prompt"] = all(
        k in table for k in tasks)
    checks["the prompt states what each engine requires"] = (
        "requires: keys, signal" in table and "requires: clock, target" in table)
    checks["the prompt states each engine's key bounds"] = (
        "keys:     2" in table and "keys:     1-4" in table)
    checks["the reply enum is the catalog, not a copy of it"] = (
        reply_schema(tasks)["properties"]["kind"]["enum"] == list(tasks))
    checks["no engine name is written into the prompt text"] = not any(
        k in SYSTEM for k in tasks)

    # ---- a draft survives being described to the model ---------------------
    split = {
        "name": "d", "version": "1",
        "metrics": [{"name": "sum_stars", "function": "sum", "measure": "stars"},
                    {"name": "conc", "function": "hhi", "measure": "stars",
                     "by": "cust_id"}],
        "dimensions": [{"name": "cust_id"}, {"name": "item_sku"}],
        "task": {"kind": "recommend", "keys": ["cust_id", "item_sku"],
                 "signal": ["sum_stars"],
                 "covariates": [{"metric": "conc", "availability": "past_only"}]},
    }
    s = _summarise(split)
    checks["a split draft keeps its kind"] = s["kind"] == "recommend"
    checks["a split draft keeps its keys"] = s["keys"] == ["cust_id", "item_sku"]
    checks["a pointer is shown as role target, whatever it is called"] = (
        s["metrics"][0]["role"] == "target")
    checks["a covariate is shown as role feature"] = (
        s["metrics"][1]["role"] == "feature")
    checks["the core's `by` is shown as the catalog's `entity`"] = (
        s["metrics"][1]["entity"] == "cust_id")

    clocked = {**split,
               "task": {"kind": "forecast", "keys": ["item_sku"],
                        "clock": {"event": "order_dt", "grain": "month"},
                        "target": ["sum_stars"],
                        "covariates": [{"metric": "conc"}]}}
    sc = _summarise(clocked)
    checks["a split draft keeps its clock"] = (
        sc["time_event"] == "order_dt" and sc["time_grain"] == "month")

    flat = {"name": "d", "kind": "cluster",
            "metrics": [{"name": "m", "function": "sum", "measure": "stars",
                         "role": "feature"}],
            "keys": [{"name": "k_cust", "from": "cust_id"},
                     {"name": "t", "via": "bin:month"}],
            "time": {"event": "order_dt", "grain": "month"},
            "dimensions": [{"name": "cust_id"}]}
    sf = _summarise(flat)
    checks["a flat draft is still read"] = (
        sf["kind"] == "cluster" and sf["time_event"] == "order_dt"
        and sf["metrics"][0]["role"] == "feature")
    checks["the clock is never shown as a key"] = sf["keys"] == ["cust_id"]

    # ---- accepting a reply -------------------------------------------------
    def apply(reply, draft=None):
        return _apply(draft if draft is not None else split, reply,
                      COLS, catalog, tasks)

    good = {"kind": "recommend",
            "metrics": [{"name": "sum_stars", "function": "sum",
                         "measure": "stars", "entity": "", "role": "target"}],
            "time_event": "", "time_grain": "", "series": "",
            "keys": ["cust_id", "item_sku"], "dimensions": [], "changes": []}
    prof, probs = apply(good)
    checks["a valid reply is accepted"] = not probs
    checks["an accepted reply names its keys"] = (
        [k.get("from") for k in prof.get("keys", [])] == ["cust_id", "item_sku"])

    def rejects(reply, needle, draft=None):
        _p, probs = apply(reply, draft)
        return bool(probs) and any(needle in x for x in probs)

    checks["a hallucinated column is rejected"] = rejects(
        {**good, "metrics": [{**good["metrics"][0], "measure": "nonesuch"}]},
        "is not a column")
    checks["a function outside the catalog is rejected"] = rejects(
        {**good, "metrics": [{**good["metrics"][0], "function": "wizardry"}]},
        "not in the catalog")
    checks["a reply predicting nothing is rejected"] = rejects(
        {**good, "metrics": [{**good["metrics"][0], "role": "feature"}]},
        "no metric has role 'target'")
    checks["an unknown engine is rejected"] = rejects(
        {**good, "kind": "telepathy"}, "is not an engine")

    # the rules that used to be a hardcoded tuple, now read off the catalog
    checks["a clock on an engine with no concept of one is rejected"] = rejects(
        {**good, "time_event": "order_dt", "time_grain": "month"},
        "has no concept of a clock")
    checks["an engine that requires a clock is held to it"] = rejects(
        {**good, "kind": "forecast", "keys": ["item_sku"]},
        "requires a clock")
    checks["the wrong key count for the engine is rejected"] = rejects(
        {**good, "keys": ["cust_id"]}, "takes 2..2 key(s)")

    # The old hardcoded pair forbade a clock to classify, cluster and regress.
    # The catalog permits all three, so this must now be ACCEPTED.
    classify_clocked = {
        "kind": "classify",
        "metrics": [{"name": "label_status", "function": "label",
                     "measure": "status", "entity": "", "role": "target"},
                    {"name": "mean_stars", "function": "mean",
                     "measure": "stars", "entity": "", "role": "feature"}],
        "time_event": "order_dt", "time_grain": "month", "series": "",
        "keys": ["cust_id"], "dimensions": [], "changes": []}
    cprof, cprobs = apply(classify_clocked)
    checks["a clock on an engine that merely permits one is accepted"] = not cprobs

    # ---- a reply that gives no keys inherits the draft's --------------------
    kept, kprobs = apply({**good, "keys": []})
    checks["an empty key list falls back to the draft's keys"] = (
        not kprobs
        and [k.get("from") for k in kept.get("keys", [])]
        == ["cust_id", "item_sku"])

    row = {"kind": "classify",
           "metrics": [{"name": "label_status", "function": "label",
                        "measure": "status", "entity": "", "role": "target"},
                       {"name": "mean_stars", "function": "mean",
                        "measure": "stars", "entity": "", "role": "feature"}],
           "time_event": "", "time_grain": "", "series": "",
           "keys": ["__row_number__"], "dimensions": [], "changes": []}
    rprof, rprobs = apply(row)
    checks["__row_number__ becomes a position key"] = (
        not rprobs and rprof["keys"] == [{"name": "k_row", "via": "row_number"}])

    # ---- the fence ---------------------------------------------------------
    keyless = {"name": "d", "metrics": split["metrics"],
               "dimensions": split["dimensions"], "task": {"kind": "recommend"}}
    _p, fprobs = apply({**good, "keys": []}, draft=keyless)
    checks["no keys anywhere is a rejection, not an empty frame"] = bool(fprobs)
    ok, why = available("")
    checks["availability is reported honestly"] = (
        isinstance(ok, bool) and bool(why))

    # ---- an accepted reply survives the rest of the pipeline ---------------
    if finalize is not None and not cprobs:
        fp, fr = finalize({"draft": cprof})
        checks["an accepted reply finalizes"] = fr["ok"]
        checks["its role lifts to the engine's own pointer"] = (
            fp["task"]["label"] == ["label_status"] and "target" not in fp["task"])
        if compile_fn is not None:
            spec, _rep = compile_fn(fp, rebind({"source": "s", "bind": {}},
                                               cprof, "s"))
            checks["and compiles to a SPEC"] = (
                spec["aggregates"][0]["name"] == "y" and len(spec["keys"]) == 2)

    passed = sum(1 for v in checks.values() if v)
    return {"service": "llm-profile", "passed": passed, "total": len(checks),
            "ok": passed == len(checks), "checks": checks}
