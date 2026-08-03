"""profile-generator: the deterministic bookend around the LLM authoring loop.

This service produces and validates the OSI-compatible *profile* -- the "what
things MEAN" document that spec-compiler later pairs with a binding and lowers
to a mapping-layer SPEC. It deliberately does NOT call an LLM. The authoring
loop is:

    schema + goal
        -> profile-generator (mode: suggest)   deterministic scaffold  <- here
        -> [ LLM enriches the draft ]           an external STP node, optional
        -> human edits + signs                  the frontend
        -> profile-generator (mode: finalize)  deterministic validate   <- here
        -> spec-compiler                        profile + binding -> SPEC

So the LLM only ever touches a draft between two deterministic, testable steps;
it is never in the path that produces the SPEC. This service is what makes the
"LLM drafts, human signs, machine compiles" split real.

Two modes:

  * suggest  -- given a column schema and a plain-language goal, scaffold a
                draft profile (pick the time column, a measure, a series grain,
                an entity, and a function keyed off the goal) plus an identity
                binding stub. A starting point for the human/LLM, not the truth.
  * finalize -- given a draft profile, validate + normalize it into a clean,
                signable profile (roles present, functions real, grain known,
                at least one target, features defaulted to past_only) and a
                report. Rejects anything the compiler would later choke on.

Input   request.parameters   mode + the inputs for that mode (see schema())
Output  RunResponse.data     [ <the profile> ]   (one row)
        RunResponse.metadata report; for suggest, also the binding stub

Deterministic, stdlib only - no LLM, no package_dependencies.
"""

import asyncio
import re
from datetime import datetime

from spl.core.base_service.base_service_class import BaseService
from spl.core.service_types import ParameterEnum

try:  # sibling modules: package-relative when deployed, flat when run locally
    from . import function_library, task_library
except ImportError:  # pragma: no cover
    import function_library
    import task_library

GRAINS = ("day", "week", "month", "quarter", "year")
ROLES = ("target", "feature")            # legacy authoring vocabulary only
AVAIL = task_library.AVAIL

# Kinds that were spelled differently before the catalog existed.
KIND_ALIASES = {"anomaly": "detect_anomaly"}

# name-shape hints used only by the suggest heuristics
_DATEISH = re.compile(r"(date|_dt$|^dt$|time|period|timestamp|^ds$|_at$|day)", re.I)
_ENTITYISH = re.compile(r"(vendor|recipient|parent|supplier|customer|company|"
                        r"account|entity|name|_id$|^id$)", re.I)
# Which questions have a time axis at all is no longer a tuple kept in step with
# the compiler's CONTRACT by hand -- it is read from the one shared catalog, so
# the service that WRITES a task and the service that LOWERS it cannot disagree
# about what a kind needs. Adding an engine is a row in task_library.csv.
CLOCKED_KINDS = tuple(k for k in task_library.kinds()
                      if "clock" in task_library.plan(k)["requires"])
CLOCKLESS_KINDS = tuple(k for k in task_library.kinds() if k not in CLOCKED_KINDS)

def task_field(profile, field, default=None):
    """Read a question-shaped field from `profile.task`, or the top level.

    The profile splits in two: `task` is what is being asked (the kind, and
    what identifies a row), and everything else describes the data. Both
    shapes are accepted — a flat profile is the original vocabulary and still
    compiles unchanged — so the split can be adopted, or backed out, without a
    migration. Deliberately NOT gated on `version`: gating catches a typo like
    "tsak" that would otherwise read as a flat profile with no kind, but it
    also commits to a schema before anyone has lived with it.
    """
    t = profile.get("task")
    if isinstance(t, dict) and field in t:
        return t[field]
    return profile.get(field, default)

KNOWN_KINDS = task_library.kinds()

_RECOMMENDISH = re.compile(
    r"(recommend|suggest.*(item|product|movie|title|track|article)|"
    r"what should|who should|similar (items?|products?)|"
    r"personalis|personaliz|cross[- ]sell|up[- ]sell|next best)", re.I)
_CLASSIFYISH = re.compile(
    r"(classif|categoris|categoriz|which (category|class|type|group|kind)|"
    r"predict (whether|if|which)|at risk|risk of|churn|fraud|spam|"
    r"flag (which|the)|label|will .* (leave|withdraw|default|fail|cancel|convert))",
    re.I)
_RANKISH = re.compile(
    r"(\brank\b|ranking|leaderboard|top[ -]?\d+|shortlist"
    r"|best \w+ for (each|every)|order .* by )", re.I)
_ANOMALYISH = re.compile(
    r"(anomal|outlier|unusual|abnormal|suspicious|deviat"
    r"|don'?t belong|doesn'?t belong|out of the ordinary|\bspike)", re.I)
_CLUSTERISH = re.compile(
    r"(cluster|segment|cohort|persona|typolog|group \w+ (together|by behaviour"
    r"|by behavior|that behave)|behave alike|look alike)", re.I)
_REGRESSISH = re.compile(
    r"((estimate|predict|project|score) (\w+\s+){0,5}(for|per) (each|every)"
    r"|how much (\w+\s+){0,5}(for|per) (each|every)"
    r"|score each|value of each)", re.I)

# Checked in this order, first match wins. Specific questions before general
# ones. Recommendation is first: "suggest which items" is a recommendation even
# though it also matches "which". Anomaly precedes classify so that "flag
# unusual spend" is not read as "flag which" -> a class. Forecasting is the
# default, because it is the only kind whose ABSENCE of a keyword is meaningful
# -- a plain "monthly sales" is a forecast.
KIND_PATTERNS = (
    ("recommend", _RECOMMENDISH),
    ("rank", _RANKISH),
    ("detect_anomaly", _ANOMALYISH),
    ("cluster", _CLUSTERISH),
    ("classify", _CLASSIFYISH),
    ("regress", _REGRESSISH),
)
DEFAULT_KIND = "forecast"

# Does the goal ask for a time axis? Only consulted for a kind that PERMITS a
# clock without requiring one -- adding a grain nobody asked for silently
# changes the question, and "segment customers" is not "segment customers per
# month".
_CLOCKISH = re.compile(
    r"(over time|trend|time series|forecast|project(ion|ed)?|season"
    r"|per (day|week|month|quarter|year)|by (day|week|month|quarter|year)"
    r"|each (day|week|month|quarter|year)|next (day|week|month|quarter|year)"
    r"|\b(daily|weekly|monthly|quarterly|yearly|annual|annually)\b)", re.I)
_DIMISH = re.compile(r"(category|type|class|region|state|store|segment|group|"
                     r"naics|sector|dept|department|channel|product|market|branch)", re.I)
_NUMTYPES = ("number", "numeric", "integer", "int", "float", "double", "decimal", "money")

# Tokens too generic to prove the goal meant a particular column.
_WEAK_TOKENS = {"id", "num", "nbr", "no", "code", "key", "col", "val", "the"}

# Formats a sample value may take. Kept in step with the mapping layer's
# DATE_FORMATS: a column this accepts must be one the executor can bucket.
_SAMPLE_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y%m%d",
                        "%Y-%m", "%Y/%m", "%m/%Y", "%m-%Y",
                        "%b %Y", "%B %Y", "%d %b %Y", "%d %B %Y")


def _looks_like_date(sample):
    """Does this value parse as a date? Names lie; values do not."""
    if sample is None:
        return False
    s = str(sample).strip()
    if not s or not any(ch.isdigit() for ch in s):
        return False
    # A plain number is a quantity, not a date - except an 8-digit YYYYMMDD.
    try:
        float(s.replace(",", ""))
        if not (len(s) == 8 and s.isdigit()):
            return False
    except ValueError:
        pass
    for fmt in _SAMPLE_DATE_FORMATS:
        try:
            datetime.strptime(s, fmt)
            return True
        except ValueError:
            continue
    return False
_DATETYPES = ("date", "datetime", "timestamp", "time")


class ProfileGeneratorService(BaseService):

    # ------------------------------------------------------------------ run

    async def _run(self, request: BaseService.RunRequest) -> BaseService.RunResponse:
        p = self._param_root(request.parameters or {})
        mode = str(p.get("mode") or "suggest").strip().lower()
        try:
            if mode == "suggest":
                profile, report = await asyncio.to_thread(self._suggest, p)
            elif mode == "finalize":
                profile, report = await asyncio.to_thread(self._finalize, p)
            else:
                raise ValueError(f"mode must be 'suggest' or 'finalize', got '{mode}'")
        except ValueError as e:
            self.logger.error(f"profile-generator ({mode}): {e}")
            return BaseService.RunResponse(data=[{"error": str(e)}])

        if report.get("warnings"):
            self.logger.warning(
                f"profile-generator ({mode}): {len(report['warnings'])} warning(s): "
                + "; ".join(report["warnings"]))
        self.logger.info(
            f"profile-generator ({mode}): profile '{profile.get('name')}' "
            f"with {len(profile.get('metrics', []))} metric(s)")
        return BaseService.RunResponse(data=[profile], metadata=report)

    @staticmethod
    def _param_root(parameters):
        inner = parameters.get("serviceInstructions")
        return inner if isinstance(inner, dict) else parameters

    # --------------------------------------------------------- suggest mode

    @staticmethod
    def triage(goal_l):
        """Which of the catalogued kinds is this goal asking for?

        Deterministic and ordered, so the same sentence always produces the
        same question. This runs in FRONT of the language model: even a total
        LLM outage still produces a correctly-shaped task.
        """
        for kind, pattern in KIND_PATTERNS:
            if pattern.search(goal_l):
                return kind
        return DEFAULT_KIND

    def _suggest(self, p):
        """Scaffold a draft, then split it into a core and a task.

        The scaffolds below all think in the flat vocabulary, because that is
        what the heuristics are about -- which column is the measure, which is
        the label. `_split` is what turns their answer into the two documents,
        in one place, so no scaffold has to remember to.
        """
        profile, report = self._suggest_flat(p)
        kind = report.get("kind") or DEFAULT_KIND
        profile = self._split(profile, kind)
        report["kind"] = kind
        report["task"] = profile["task"]
        return profile, report

    @staticmethod
    def _split(profile, kind):
        """A flat draft -> a core and a task.

        One question sorts every field: with no engine attached, does this
        still describe something real about the data? A metric's function and
        measure do; its `role` does not, because "which column is predicted" is
        meaningless until something is predicting. So the role comes off the
        metric and the task gains a POINTER to it.

        That inversion is what freezes the metric schema. A role stamped on a
        metric grows its enum every time an engine is added -- a recommender
        wants `signal`, a classifier wants `label`. A pointer does not.
        """
        plan = task_library.plan(kind)
        pointer = task_library.pointer_of(plan) or "target"
        task = {"kind": kind}

        time = profile.get("time")
        if isinstance(time, dict) and str(time.get("event") or "").strip():
            clock = {"event": str(time["event"]).strip()}
            grain = str(time.get("grain") or "").strip().lower()
            if grain:
                clock["grain"] = grain
            arrival = str(time.get("arrival") or "").strip()
            if arrival:
                clock["arrival"] = arrival
            task["clock"] = clock

        keys = []
        raw = profile.get("keys")
        if isinstance(raw, list) and raw:
            for k in raw:
                if isinstance(k, str) and k.strip():
                    keys.append(k.strip())
                elif isinstance(k, dict):
                    frm = str(k.get("from") or "").strip()
                    via = str(k.get("via") or "").strip()
                    if via.startswith("bin:"):
                        continue                      # that is the clock
                    # A key that supplies its own value still identifies a row:
                    # `row_number` has nothing to group by, and dropping it here
                    # would tell the catalog the question had no keys at all.
                    if frm or via:
                        keys.append(frm or via)
        else:
            xdeep = profile.get("x-deep") or profile.get("x_deep") or {}
            series = str(xdeep.get("series") or "").strip() if isinstance(xdeep, dict) else ""
            if series:
                keys.append(series)
        if keys:
            task["keys"] = keys

        core_metrics, pointed, covariates = [], [], []
        for m in (profile.get("metrics") or []):
            if not isinstance(m, dict):
                continue
            core_metrics.append({k: v for k, v in m.items()
                                 if k not in ("role", "availability")})
            name = str(m.get("name") or "").strip()
            if str(m.get("role") or "").strip().lower() == "target":
                pointed.append(name)
            else:
                covariates.append({
                    "metric": name,
                    "availability": str(m.get("availability") or "").strip().lower()
                    or "past_only"})
        if pointed:
            task[pointer] = pointed
        if covariates:
            task["covariates"] = covariates

        out = {"name": profile.get("name"), "version": profile.get("version", "1"),
               "datasets": profile.get("datasets") or [{"name": "dataset"}],
               "metrics": core_metrics,
               "dimensions": profile.get("dimensions") or []}
        for extra in ("filters", "keys"):
            if profile.get(extra):
                out[extra] = profile[extra]
        out["task"] = task
        return out

    def _suggest_flat(self, p):
        cols = self._read_schema(p.get("schema"))
        if not cols:
            raise ValueError("suggest needs a schema (a non-empty list of columns)")
        goal = str(p.get("goal") or "").strip()
        goal_l = goal.lower()
        dataset = str(p.get("dataset") or "dataset").strip() or "dataset"
        warnings = []

        # -- which question is this? -------------------------------------
        # Deciding this first, before demanding a date column, is what stops an
        # entirely valid interaction log being rejected as "not forecastable".
        kind = str(p.get("kind") or "").strip().lower()
        kind = KIND_ALIASES.get(kind, kind)
        if kind:
            task_library.plan(kind, "kind")          # named, so it must exist
        else:
            kind = self.triage(goal_l)
        if kind in ("recommend", "rank"):
            return self._suggest_recommend(cols, goal, goal_l, dataset, warnings, kind)
        if kind == "classify":
            return self._suggest_classify(cols, goal, goal_l, dataset, warnings)
        if kind in ("cluster", "regress"):
            return self._suggest_entity(cols, goal, goal_l, dataset, warnings, kind)

        # -- the clock ----------------------------------------------------
        # A forecast is meaningless without one. `detect_anomaly` reads more
        # naturally over time, but the catalog only PERMITS it a clock, so an
        # outlier hunt across entities is a legitimate question and falls
        # through to the entity-keyed scaffold rather than being refused.
        date_col = self._pick(cols, self._is_date)
        if not date_col:
            if kind not in CLOCKED_KINDS:
                return self._suggest_entity(cols, goal, goal_l, dataset, warnings, kind)

            # This used to raise, and a raise here VETOES THE LANGUAGE MODEL.
            # The scaffold always runs first and the LLM stage corrects its
            # draft, so refusing at this point halts the pipeline before the
            # model ever reads the sentence. `triage` reads "predict who will
            # survive" as a forecast; the file has no clock; the run died --
            # while the model, given any draft at all, names the label
            # immediately. A keyword regex must not decide whether the reader
            # gets to read.
            #
            # So demote to the clockless kind the data can actually support and
            # say so. A wrong guess is one stage from being fixed; a refusal is
            # not fixable at all.
            text_lab, num_lab = self._label_candidates(cols)
            demoted = "classify" if (text_lab or num_lab) else "regress"
            warnings.append(
                f"the goal reads as a '{kind}', but this file has no date "
                f"column, so a clock is impossible; drafted a '{demoted}' "
                f"instead -- say what you want in the goal")
            if demoted == "classify":
                return self._suggest_classify(cols, goal, goal_l, dataset, warnings)
            return self._suggest_entity(cols, goal, goal_l, dataset, warnings,
                                        demoted)

        # -- grain: explicit override, else goal keyword, else month -----
        grain = str(p.get("grain") or "").strip().lower()
        if grain and grain not in GRAINS:
            raise ValueError(f"grain must be one of {GRAINS}, got '{grain}'")
        if not grain:
            grain = next((g for g in GRAINS if g in goal_l
                          or (g + "ly") in goal_l), "month")

        # -- function keyed off the goal ---------------------------------
        fn = self._fn_from_goal(goal_l)
        lib = function_library.catalog()
        entry = lib.get(fn) or {}
        needs_measure = "measure" in entry.get("needs", []) or fn not in ("count",)
        is_group = entry.get("kind") == "group"

        # -- measure: a numeric column, preferring one named in the goal -
        measure_col = None
        if needs_measure:
            nums = [c for c in cols if self._is_number(c) and not self._is_date(c)]
            measure_col = next((c["name"] for c in nums
                                if c["name"].lower() in goal_l), None)
            if not measure_col and nums:
                measure_col = nums[0]["name"]
            if not measure_col:
                warnings.append(
                    f"no numeric column for '{fn}'; falling back to count")
                fn, needs_measure, is_group = "count", False, False

        # -- entity for a group function ---------------------------------
        entity_col = None
        if is_group:
            entity_col = self._pick(cols, self._is_entity, exclude={date_col, measure_col})
            if not entity_col:
                warnings.append(
                    f"'{fn}' needs an entity to form shares but none was found; "
                    f"falling back to sum")
                fn, is_group = "sum", False

        # -- series grain (a low-cardinality dimension) ------------------
        series_col = (self._pick(cols, self._is_dim,
                                 exclude={date_col, measure_col, entity_col})
                      or self._pick(cols, self._is_plain_string,
                                    exclude={date_col, measure_col, entity_col}))
        if not series_col:
            warnings.append("no dimension found for the series grain; "
                            "forecasting a single global series")

        # -- assemble the draft profile (logical names = the columns) ----
        metric_name = {"hhi": "concentration", "top_share": "top_share",
                       "count": "record_count"}.get(fn, f"{fn}_{measure_col}")
        metric = {"name": metric_name, "function": fn, "role": "target"}
        if measure_col:
            metric["measure"] = measure_col
        if entity_col:
            metric["by"] = entity_col

        dims = [c for c in (series_col, entity_col) if c]
        profile = {
            "name": self._slug(goal) or dataset,
            "version": "1",
            "datasets": [{"name": dataset}],
            "metrics": [metric],
            "dimensions": [{"name": d} for d in dims],
            "time": {"event": date_col, "grain": grain},
        }
        if series_col:
            profile["x-deep"] = {"series": series_col}

        # -- identity binding stub: logical == physical, ready to edit ---
        logicals = [date_col] + ([measure_col] if measure_col else []) \
            + ([entity_col] if entity_col else []) + ([series_col] if series_col else [])
        binding_stub = {
            "source": dataset,
            "bind": {name: name for name in dict.fromkeys(logicals)},
            "available_from": "",
        }

        report = {
            "mode": "suggest",
            "goal": goal,
            "kind": kind,
            "chose": {
                "time": date_col, "grain": grain, "function": fn,
                "measure": measure_col, "entity": entity_col, "series": series_col,
            },
            "binding_stub": binding_stub,
            "warnings": warnings,
            "note": ("A deterministic scaffold from column names + the goal. Hand to "
                     "the LLM to enrich, then a human edits and signs. Not the truth."),
        }
        return profile, report

    # --------------------------------------------------- suggest: entity-keyed

    def _suggest_entity(self, cols, goal, goal_l, dataset, warnings, kind):
        """One row per entity: cluster and regress.

        Neither has a grid to build or a class to find, so the shape is the
        simplest one in the catalog — key on the thing, measure everything else
        about it. What separates them is only whether anything is predicted:
        a regression names a target, a clustering names nothing at all and
        carries every metric alongside.
        """
        entity = (self._pick(cols, self._is_entity)
                  or self._pick(cols, self._is_dim)
                  or self._pick(cols, self._is_plain_string))
        # A column the goal names by hand outranks any shape heuristic, so
        # "segment CUSTOMERS" keys on cust_id rather than on whatever sits
        # first and looks dimension-shaped.
        named = next((c["name"] for c in cols
                      if not self._is_date(c)
                      and self._goal_names(c["name"], goal_l)
                      and (self._is_entity(c) or self._is_dim(c)
                           or self._is_plain_string(c))), None)
        entity = named or entity
        if not entity:
            raise ValueError(
                f"a '{kind}' keys on one entity, but no column in the schema "
                f"looks like one (an id, a name, or a low-cardinality category)")

        # A clock is permitted here but never assumed: adding a grain nobody
        # asked for turns "segment customers" into a different question.
        date_col = (self._pick(cols, self._is_date)
                    if _CLOCKISH.search(goal_l) else None)
        grain = (next((g for g in GRAINS if g in goal_l or (g + "ly") in goal_l),
                      "month") if date_col else "")

        nums = [c["name"] for c in cols
                if self._is_number(c) and not self._is_date(c) and c["name"] != entity]
        if not nums:
            raise ValueError(
                f"a '{kind}' needs something measurable about '{entity}', but the "
                f"schema has no numeric column")
        fn = self._fn_from_goal(goal_l)
        if (function_library.catalog().get(fn) or {}).get("kind") == "group":
            fn = "sum"                       # no entity to form shares against

        pointer = task_library.pointer_of(task_library.plan(kind))
        metrics = []
        for i, col in enumerate(nums):
            # For a regression the first numeric is what gets predicted, unless
            # the goal named one; everything else rides along.
            metrics.append({"name": f"{fn}_{col}", "function": fn, "measure": col,
                            "role": "feature"})
        if pointer:
            pick = next((m for m in metrics
                         if self._goal_names(m["measure"], goal_l)), metrics[0])
            pick["role"] = "target"
        else:
            warnings.append(
                f"a '{kind}' predicts nothing, so every metric is carried "
                f"alongside rather than named as a target")

        profile = {
            "name": self._slug(goal) or dataset,
            "version": "1",
            "datasets": [{"name": dataset}],
            "metrics": metrics,
            "dimensions": [{"name": entity}],
            # The short form: one name, so the compiler's own naming rule
            # decides the output column. Spelling it out here would fix the
            # column name before anyone has chosen a dialect.
            "keys": [entity],
        }
        if date_col:
            profile["time"] = {"event": date_col, "grain": grain}

        logicals = [entity] + nums + ([date_col] if date_col else [])
        report = {
            "mode": "suggest", "goal": goal, "kind": kind,
            "chose": {"entity": entity, "function": fn, "measures": nums,
                      "time": date_col, "grain": grain or None,
                      "predicted": next((m["name"] for m in metrics
                                         if m["role"] == "target"), None)},
            "binding_stub": {"source": dataset,
                             "bind": {n: n for n in dict.fromkeys(logicals)},
                             "available_from": ""},
            "warnings": warnings,
            "note": (f"A '{kind}' keyed on '{entity}'. Which column identifies the "
                     f"thing being grouped is a guess from names and order — check "
                     f"it before signing."),
        }
        return profile, report

    # ------------------------------------------------------- suggest: classify

    def _label_candidates(self, cols):
        """Columns that could plausibly hold a class label, best first.

        Extracted so the classify scaffold and the no-clock demotion ask the
        same question. Two rules that drifted apart would be worse than either:
        the demotion would offer a classification the scaffold then could not
        build.

        Text first, then numerics whose VALUE SHAPE looks like a class id -- a
        whole number from a small run. 71.78 is never a class, however few times
        it appears, and counting distinct values conflated the two.
        """
        def few_values(c, hi):
            n = c.get("cardinality")
            return isinstance(n, (int, float)) and 1 < n <= hi

        def looks_like_a_class(c):
            sample = str(c.get("sample") or "").strip()
            if not sample:
                return False
            try:
                v = float(sample)
            except ValueError:
                return False
            return v == int(v) and abs(v) < 1000

        text = [c for c in cols
                if not self._is_date(c) and not self._is_number(c)
                and few_values(c, 20)]
        numeric = [c for c in cols
                   if self._is_number(c) and not self._is_date(c)
                   and few_values(c, 20) and looks_like_a_class(c)]
        return text, numeric

    def _suggest_classify(self, cols, goal, goal_l, dataset, warnings):
        """A labelled table: one row per thing, one categorical column to predict.

        Deliberately a weak draft rather than a clever one. Mapping "at risk of
        withdrawing" to a column called WITHDRAWN_FLAG is exactly the judgment a
        keyword table cannot make and a model can, so this picks a defensible
        floor and leaves the reading to the LLM stage — the same division that
        stopped `market fragmentation` needing a synonym entry for `hhi`.
        """
        # The label: a categorical column with few distinct values. A column the
        # goal actually names wins, because that is the user being explicit.
        cands, numeric_cands = self._label_candidates(cols)
        named = [c for c in (cands + numeric_cands) if c["name"].lower() in goal_l]
        pool = cands or numeric_cands
        if named:
            label = named[0]
        elif pool:
            label = min(pool, key=lambda c: (c["cardinality"], c["name"]))
            if pool is numeric_cands:
                warnings.append(
                    f"'{label['name']}' is numeric but its values look like "
                    f"class ids ({int(label['cardinality'])} of them), so it "
                    f"was read as a label rather than something to measure")
            # A tie between look-alike columns is a coin flip, and saying so is
            # the difference between a draft you check and one you trust.
            rivals = [c["name"] for c in pool
                      if c["name"] != label["name"]
                      and c["cardinality"] == label["cardinality"]]
            if rivals:
                warnings.append(
                    f"'{label['name']}' was picked as the label, but "
                    f"{', '.join(repr(r) for r in rivals[:3])} "
                    f"{'is' if len(rivals) == 1 else 'are'} equally "
                    f"label-shaped — name the one you mean in the goal")
        else:
            # Still no candidate: emit the weakest possible draft rather than
            # refusing, so the LLM stage gets something to correct. The warning
            # is the honest signal; a raise here would end the run.
            fallback = next((c for c in reversed(cols)
                             if not self._is_date(c)), cols[-1])
            label = fallback
            warnings.append(
                f"no column looks like a class label; fell back to "
                f"'{label['name']}' (the last non-date column). This is a "
                f"guess — say which column holds the label in the goal")

        feats = [c for c in cols
                 if self._is_number(c) and not self._is_date(c)
                 and c["name"] != label["name"]]
        if not feats:
            raise ValueError(
                f"no numeric columns to learn from — '{label['name']}' looks "
                f"like a label but nothing measurable accompanies it")

        # The key: a real identifier if the file has one, otherwise each row is
        # already the thing being classified and its position is its identity.
        ident = next((c for c in cols
                      if c["name"] != label["name"]
                      and _ENTITYISH.search(c["name"])
                      and isinstance(c.get("cardinality"), (int, float))
                      and c["cardinality"] > 1), None)
        if ident:
            keys = [{"name": f"k_{ident['name']}", "from": ident["name"]}]
            dims = [{"name": ident["name"]}]
        else:
            keys = [{"name": "k_row", "via": "row_number"}]
            dims = []
            warnings.append(
                "no id column found, so each input row becomes one frame row "
                "(key 'row_number'); if rows should be grouped, name the "
                "column that groups them")

        metrics = [{"name": f"label_{label['name']}", "function": "label",
                    "role": "target", "measure": label["name"]}]
        for c in feats:
            metrics.append({"name": f"mean_{c['name']}", "function": "mean",
                            "role": "feature", "measure": c["name"]})

        profile = {
            "name": self._slug(goal) or dataset,
            "version": "1",
            "kind": "classify",
            "datasets": [{"name": dataset}],
            "keys": keys,
            "metrics": metrics,
            "dimensions": dims,
        }
        bound = [label["name"]] + [c["name"] for c in feats]
        if ident:
            bound.append(ident["name"])
        binding_stub = {"source": dataset,
                        "bind": {n: n for n in dict.fromkeys(bound)},
                        "available_from": ""}
        report = {
            "mode": "suggest", "goal": goal, "kind": "classify",
            "chose": {"label": label["name"],
                      "classes": label.get("cardinality"),
                      "key": ident["name"] if ident else "row_number",
                      "features": [c["name"] for c in feats]},
            "binding_stub": binding_stub,
            "warnings": warnings,
            "note": ("The label is the non-numeric column with the fewest "
                     "distinct values, and every other numeric column is a "
                     "feature. That is a floor, not a reading of the goal — "
                     "check it, or let the model correct it."),
        }
        return profile, report

    # ------------------------------------------------------ suggest: recommend

    def _suggest_recommend(self, cols, goal, goal_l, dataset, warnings,
                           kind="recommend"):
        """A user x item grid. Two entity keys, one weight, no clock.

        `rank` is the same frame asked a different question -- order the
        items within a query key rather than score every pair -- so it is the
        same scaffold with a different row of the catalog behind it.
        """
        # An id-shaped column beats a plain category: `viewer` and `title` are
        # the grid, `genre` is a description of one axis of it.
        ranked = [c for c in cols if self._is_entity(c) or self._is_dim(c)]
        ranked += [c for c in cols if self._is_plain_string(c) and c not in ranked]
        # An id is often written as a number. `store_nbr` holds 12 and 44, so
        # every entity test rejected it and a two-axis grid came back with one
        # axis. A low-cardinality numeric with an id-shaped name is a thing,
        # not a quantity — the same trap as a NAICS code being averaged.
        ranked += [c for c in cols
                   if c not in ranked and self._is_number(c)
                   and not self._is_date(c)
                   and (_ENTITYISH.search(c["name"]) or _DIMISH.search(c["name"]))
                   and isinstance(c.get("cardinality"), (int, float))
                   and 1 < c["cardinality"] <= 50]
        named = [c for c in ranked if c["name"].lower() in goal_l]
        for c in reversed(named):                       # the goal names them first
            ranked.remove(c)
            ranked.insert(0, c)
        if len(ranked) < 2:
            raise ValueError(
                f"a '{kind}' needs two entity columns (who, and what) — "
                f"found {len(ranked)} in the schema")
        user_col, item_col = ranked[0]["name"], ranked[1]["name"]

        # The weight is what engagement was worth. Absent one, every observed
        # pair counts once, which is the implicit-feedback reading.
        nums = [c for c in cols if self._is_number(c) and not self._is_date(c)
                and c["name"] not in (user_col, item_col)]
        weight = next((c["name"] for c in nums if c["name"].lower() in goal_l),
                      nums[0]["name"] if nums else None)
        if weight:
            metric = {"name": f"max_{weight}", "function": "max",
                      "role": "target", "measure": weight}
        else:
            metric = {"name": "interactions", "function": "count",
                      "role": "target"}
            warnings.append(
                "no numeric column to weight the grid; counting interactions "
                "instead, which is the implicit-feedback reading")

        profile = {
            "name": self._slug(goal) or dataset,
            "version": "1",
            "kind": kind,
            "datasets": [{"name": dataset}],
            "keys": [{"name": f"k_{user_col}", "from": user_col},
                     {"name": f"k_{item_col}", "from": item_col}],
            "metrics": [metric],
            "dimensions": [{"name": user_col}, {"name": item_col}],
        }
        binding_stub = {
            "source": dataset,
            "bind": {n: n for n in dict.fromkeys(
                [user_col, item_col] + ([weight] if weight else []))},
            "available_from": "",
        }
        report = {
            "mode": "suggest", "goal": goal, "kind": kind,
            "chose": {"user": user_col, "item": item_col, "weight": weight,
                      "function": metric["function"]},
            "binding_stub": binding_stub,
            "warnings": warnings,
            "note": ("A user x item grid keyed on two entity columns, with no "
                     "clock. Which column is the user and which is the item is "
                     "a guess from names and order — check it before signing."),
        }
        return profile, report

    # -------------------------------------------------------- finalize mode

    def _finalize(self, p):
        draft = p.get("draft") or p.get("profile")
        if not isinstance(draft, dict):
            raise ValueError("finalize needs a 'draft' profile object")
        warnings = []
        lib = function_library.catalog()

        metrics = draft.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise ValueError("draft.metrics must be a non-empty list")

        # Only a question with a clock needs one. Demanding `time.event` of
        # every profile rejected recommendation outright, which keys a user x
        # item grid and has no time axis at all.
        kind = str(task_field(draft, "kind") or DEFAULT_KIND).strip().lower()
        kind = KIND_ALIASES.get(kind, kind)
        # The clock may be spelled either way: `time` on a flat draft, or
        # `task.clock` on a split one. Reading both is what lets finalize run
        # over its own output.
        time = draft.get("time")
        if not (isinstance(time, dict) and str(time.get("event") or "").strip()):
            native_clock = task_field(draft, "clock")
            if isinstance(native_clock, dict):
                time = native_clock
        has_time = isinstance(time, dict) and str(time.get("event") or "").strip()
        if kind in CLOCKED_KINDS and not has_time:
            raise ValueError(
                f"draft.time.event is required for kind '{kind}' "
                f"(kinds without a clock: "
                f"{', '.join(sorted(k for k in CLOCKLESS_KINDS if k != kind))})")
        grain = ""
        if has_time:
            grain = str(time.get("grain") or "").strip().lower()
            if grain not in GRAINS:
                raise ValueError(
                    f"draft.time.grain must be one of {GRAINS}, got '{time.get('grain')}'")

        dim_names = []
        for d in (draft.get("dimensions") or []):
            if isinstance(d, dict) and str(d.get("name") or "").strip():
                nm = str(d["name"]).strip()
                if nm not in dim_names:
                    dim_names.append(nm)

        # Does the draft already say what is being asked, or is it still
        # spelled as roles on the metrics?
        native = draft.get("task") if isinstance(draft.get("task"), dict) else None
        plan = task_library.plan(kind, "draft.kind")
        pointer = task_library.pointer_of(plan) or "target"
        # "Authored" means the task already says what is predicted or carried.
        # A clustering says it by naming covariates and no pointer at all --
        # that is a complete answer, not a missing one.
        authored = bool(native) and (
            any(f in native for f in task_library.POINTERS)
            or "covariates" in native)

        norm_metrics = []
        legacy_roles, legacy_avail = {}, {}
        n_targets = 0
        for i, m in enumerate(metrics):
            if not isinstance(m, dict):
                raise ValueError(f"metrics[{i}] must be an object")
            name = str(m.get("name") or "").strip()
            if not name:
                raise ValueError(f"metrics[{i}] needs a name")
            fn = str(m.get("function") or "").strip()
            entry = lib.get(fn)
            if entry is None:
                raise ValueError(
                    f"metric '{name}' uses unknown function '{fn}' "
                    f"(not in the library: {sorted(lib)})")
            # A role is demanded only while the task does not already say what
            # is predicted. Once it does, a role on the metric is redundant --
            # and the point of the split is that it stops being the metric's
            # business at all.
            role = str(m.get("role") or "").strip().lower()
            if not authored and role not in ROLES:
                raise ValueError(
                    f"metric '{name}' needs role one of {ROLES} "
                    f"(or name it under task.{pointer} / task.covariates)")

            nm = {"name": name, "function": fn}
            by = str(m.get("by") or m.get("entity") or "").strip()
            if entry.get("kind") == "group":
                if not by:
                    raise ValueError(
                        f"metric '{name}' uses group function '{fn}' but has no 'by' entity")
                nm["by"] = by
                if by not in dim_names:
                    warnings.append(f"metric '{name}' groups by '{by}', "
                                    f"which is not listed under dimensions")
            elif by:
                warnings.append(f"metric '{name}' has 'by: {by}' but '{fn}' is not a "
                                f"group function; dropping it")
            measure = str(m.get("measure") or "").strip()
            if measure:
                nm["measure"] = measure
            elif fn != "count":
                warnings.append(f"metric '{name}' function '{fn}' usually needs a measure")

            # role and availability come OFF the metric here. They describe the
            # question, not the quantity, so they are remembered only long
            # enough to build the task out of them.
            if role:
                legacy_roles[name] = role
            if role == "target":
                n_targets += 1
            elif role:
                avail = str(m.get("availability") or "").strip().lower()
                if avail and avail not in AVAIL:
                    raise ValueError(
                        f"feature '{name}' availability must be one of {AVAIL}, got '{avail}'")
                if not avail:
                    avail = "past_only"
                    warnings.append(f"feature '{name}' had no availability; "
                                    f"defaulted to past_only")
                legacy_avail[name] = avail
            norm_metrics.append(nm)

        xdeep = draft.get("x-deep") or draft.get("x_deep") or {}
        series = str(xdeep.get("series") or "").strip() if isinstance(xdeep, dict) else ""
        if series and series not in dim_names:
            warnings.append(f"x-deep.series '{series}' is not listed under dimensions")

        profile = {
            "name": str(draft.get("name") or "").strip() or "profile",
            "version": str(draft.get("version") or "1").strip(),
            "datasets": draft.get("datasets") or [{"name": "dataset"}],
            "metrics": norm_metrics,
            "dimensions": [{"name": d} for d in dim_names],
        }
        if draft.get("filters"):
            profile["filters"] = draft["filters"]

        # -- the task ------------------------------------------------------
        # Always stated, never implied. `forecast` is the compiler's default, so
        # omitting it compiled identically — but the profile is the document a
        # human reads and signs, and absence there is ambiguous: it cannot be
        # told apart from a kind that failed to be set, and it leaves nothing to
        # edit when you want to change the question.
        task = {"kind": kind}
        if has_time:
            clock = {"event": str(time["event"]).strip()}
            if grain:
                clock["grain"] = grain
            arrival = str(time.get("arrival") or "").strip()
            if arrival:
                clock["arrival"] = arrival
            task["clock"] = clock
        elif isinstance(native, dict) and isinstance(native.get("clock"), dict):
            task["clock"] = self._norm_clock(native["clock"], "draft.task.clock")

        # An explicit keys[] is the general form and must survive finalize —
        # it is the only way a clockless question names its axes. The task
        # carries the dimension NAMES for the catalog to check; the objects, if
        # the author wrote any, stay at the top level where they still name the
        # output columns.
        # A top-level keys[] outranks task.keys, the same precedence the
        # compiler uses. Reading task.keys first dropped the authored objects
        # on finalize's own output — and with them the only spelling a
        # `row_number` key has, which is how a classification on data with no
        # id column identifies its rows.
        explicit_keys = draft.get("keys")
        draft_keys = (explicit_keys if isinstance(explicit_keys, list) and explicit_keys
                      else task_field(draft, "keys"))
        key_names = []
        if isinstance(draft_keys, list) and draft_keys:
            for k in draft_keys:
                if isinstance(k, str) and k.strip():
                    key_names.append(k.strip())
                elif isinstance(k, dict):
                    frm = str(k.get("from") or "").strip()
                    via = str(k.get("via") or "").strip()
                    if via.startswith("bin:"):
                        continue
                    if frm or via:
                        key_names.append(frm or via)
            if any(isinstance(k, dict) for k in draft_keys):
                profile["keys"] = draft_keys
        elif series:
            key_names.append(series)
        if key_names:
            task["keys"] = key_names

        if authored:
            for f in task_library.POINTERS:
                if native.get(f) not in (None, "", []):
                    task[f] = native[f]
            if native.get("covariates"):
                task["covariates"] = native["covariates"]
        else:
            pointed = [n for n in legacy_roles if legacy_roles[n] == "target"]
            covariates = [{"metric": n, "availability": legacy_avail.get(n, "past_only")}
                          for n in legacy_roles if legacy_roles[n] != "target"]
            if pointed:
                task[pointer] = pointed
            if covariates:
                task["covariates"] = covariates
        for k in (native or {}):
            if k != "kind" and k not in task_library.TASK_FIELDS:
                raise ValueError(
                    f"draft.task field '{k}' is not part of the task vocabulary "
                    f"({', '.join(task_library.TASK_FIELDS)})")
        profile["task"] = task

        # The catalog is the contract: a required field missing, a pointer at a
        # metric nobody declared, or a field this kind has no concept of.
        # Checking here rather than only in the compiler is what makes finalize
        # worth running — it is the last step before a human signs.
        # A self-supplying key names a derive, not a dimension, so it can never
        # appear under `dimensions`.
        check_dims = dim_names
        if check_dims and isinstance(draft_keys, list):
            check_dims = check_dims + [
                str(k.get("via") or "").strip() for k in draft_keys
                if isinstance(k, dict) and k.get("via") and not k.get("from")]
        _, kind, pointer, pointed = task_library.check(
            task, [m["name"] for m in norm_metrics], check_dims)

        carried = {str(c.get("metric") or "").strip()
                   for c in (task.get("covariates") or []) if isinstance(c, dict)}
        for m in norm_metrics:
            if m["name"] not in pointed and m["name"] not in carried:
                warnings.append(
                    f"metric '{m['name']}' is declared but the task never names it; "
                    f"spec-compiler will reject that")

        report = {
            "mode": "finalize",
            "ok": True,
            "kind": kind,
            "metrics": len(norm_metrics),
            "pointer": pointer,
            "predicted": pointed,
            "covariates": sorted(carried),
            "lifted_from_flat": not authored,
            "targets": len(pointed),
            "features": len(carried),
            "warnings": warnings,
            "note": ("Validated + normalized against the task library. Safe to sign "
                     "and hand to spec-compiler (which re-checks the binding)."),
        }
        return profile, report

    @staticmethod
    def _norm_clock(raw, where):
        event = str(raw.get("event") or "").strip()
        if not event:
            raise ValueError(f"{where}.event is required")
        grain = str(raw.get("grain") or "").strip().lower()
        if grain and grain not in GRAINS:
            raise ValueError(f"{where}.grain must be one of {GRAINS}, got '{raw.get('grain')}'")
        out = {"event": event}
        if grain:
            out["grain"] = grain
        arrival = str(raw.get("arrival") or "").strip()
        if arrival:
            out["arrival"] = arrival
        return out

    # ---------------------------------------------------- schema heuristics

    @staticmethod
    def _read_schema(schema):
        """Accept ['col', ...] or [{'name':..,'type':..,'cardinality':..}, ...]."""
        out = []
        for c in (schema or []):
            if isinstance(c, str) and c.strip():
                out.append({"name": c.strip(), "type": "", "cardinality": None})
            elif isinstance(c, dict) and str(c.get("name") or "").strip():
                out.append({
                    "name": str(c["name"]).strip(),
                    "type": str(c.get("type") or "").strip().lower(),
                    "cardinality": c.get("cardinality"),
                    "sample": c.get("sample"),
                })
        return out

    @staticmethod
    def _pick(cols, pred, exclude=frozenset()):
        for c in cols:
            if c["name"] in exclude:
                continue
            if pred(c):
                return c["name"]
        return None

    @staticmethod
    def _goal_names(col_name, goal_l):
        """Did the goal name this column, in so many words?

        Either outright, or by a word of it: "segment customers" names
        `cust_id`. Generic fragments are ignored, so `_id` alone proves nothing.
        """
        name = str(col_name).lower()
        if name and name in goal_l:
            return True
        toks = [t for t in re.split(r"[^a-z0-9]+", name)
                if len(t) > 2 and t not in _WEAK_TOKENS]
        return any(t in goal_l for t in toks)

    def _is_date(self, c):
        if c["type"] in _DATETYPES:
            return True
        if c["type"] in _NUMTYPES:
            return False
        if _DATEISH.search(c["name"]):
            return True
        # The name is only a hint. A column called `Month` holding "1949-01" is
        # a date whatever it is called, and reading the value is the difference
        # between recognising a file and rejecting it.
        return _looks_like_date(c.get("sample"))

    def _is_number(self, c):
        if c["type"] in _NUMTYPES:
            return True
        if c["type"] in _DATETYPES:
            return False
        s = c.get("sample")
        if s is not None:
            try:
                float(str(s).replace(",", ""))
                return True
            except ValueError:
                return False
        return False

    def _is_entity(self, c):
        return (not self._is_date(c) and not self._is_number(c)
                and bool(_ENTITYISH.search(c["name"])))

    def _is_dim(self, c):
        if self._is_date(c) or self._is_number(c) or self._is_entity(c):
            return False
        if _DIMISH.search(c["name"]):
            return True
        card = c.get("cardinality")
        return isinstance(card, (int, float)) and 1 < card <= 50

    def _is_plain_string(self, c):
        return not self._is_date(c) and not self._is_number(c)

    @staticmethod
    def _fn_from_goal(goal_l):
        if re.search(r"concentrat|hhi|herfindahl", goal_l):
            return "hhi"
        if re.search(r"top[ _-]?share|largest share|dominant|biggest share", goal_l):
            return "top_share"
        if re.search(r"\bcount\b|number of|how many", goal_l):
            return "count"
        if re.search(r"average|\bmean\b|\bavg\b", goal_l):
            return "mean"
        if re.search(r"\bpeak\b|highest|maximum|\bmax\b", goal_l):
            return "max"
        if re.search(r"lowest|minimum|\bmin\b", goal_l):
            return "min"
        return "sum"

    @staticmethod
    def _slug(text):
        s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
        return s[:48]

    # -------------------------------------------------------------- schema

    async def schema(self) -> dict | None:
        SSP = BaseService.ServiceSchemaProperty
        return SSP(
            key="serviceInstructions",
            type="object",
            description=("Produce or validate an OSI-compatible forecasting profile. "
                         "No LLM: 'suggest' scaffolds a draft from a schema + goal; "
                         "'finalize' validates + normalizes a draft into a signable profile."),
            properties=[
                SSP(key="mode", type="enum", default="suggest",
                    description="suggest = scaffold a draft; finalize = validate + normalize a draft",
                    enum=ParameterEnum(
                        values=["suggest", "finalize"],
                        labels={"suggest": "Suggest a draft (from schema + goal)",
                                "finalize": "Finalize (validate + normalize a draft)"})),
                # ---- suggest inputs ----
                SSP(key="schema", type="array",
                    description="suggest: the source's columns",
                    properties=[
                        SSP(key="name", type="string", required=True,
                            description="Column name"),
                        SSP(key="type", type="string",
                            description="Optional type hint (date/number/string/...)"),
                        SSP(key="cardinality", type="number",
                            description="Optional distinct-value count (helps spot dimensions)"),
                        SSP(key="sample", type="string",
                            description="Optional sample value (used to infer numeric)"),
                    ]),
                SSP(key="goal", type="string",
                    description="suggest: plain-language goal, e.g. 'quarterly supplier concentration by NAICS'"),
                SSP(key="dataset", type="string",
                    description="suggest: dataset/source name (default 'dataset')"),
                SSP(key="grain", type="enum",
                    description="suggest: force a grain (else inferred from the goal)",
                    enum=ParameterEnum(values=list(GRAINS))),
                # ---- finalize input ----
                SSP(key="draft", type="object",
                    description="finalize: the draft profile object to validate + normalize"),
            ],
        )

    # ------------------------------------------------------------ self-test

    def self_test(self):
        """Offline check: suggest on procurement + retail schemas, and finalize
        normalization + validation errors."""
        checks = {}

        # ---- suggest: procurement concentration ----
        proc = {
            "goal": "quarterly supplier concentration by NAICS category",
            "dataset": "usaspending_awards",
            "schema": [
                {"name": "action_dt", "type": "date"},
                {"name": "dollars_obligated", "type": "number"},
                {"name": "recipient_parent", "type": "string", "cardinality": 5000},
                {"name": "naics_code", "type": "string", "cardinality": 40},
                {"name": "award_type", "type": "string", "cardinality": 6},
            ],
        }
        prof, rep = self._suggest(proc)
        m, task = prof["metrics"][0], prof["task"]
        checks["suggest picks hhi from 'concentration'"] = m["function"] == "hhi"
        checks["suggest infers quarter grain"] = task["clock"]["grain"] == "quarter"
        checks["suggest binds the clock to the date col"] = task["clock"]["event"] == "action_dt"
        checks["suggest picks the numeric measure"] = m["measure"] == "dollars_obligated"
        checks["suggest picks an entity for hhi"] = m["by"] == "recipient_parent"
        checks["suggest keys on a dimension"] = task["keys"] == ["naics_code"]
        checks["suggest emits an identity binding stub"] = (
            rep["binding_stub"]["bind"].get("action_dt") == "action_dt")
        # the core says how to compute; the task says what is asked of it
        checks["a forecast names a target"] = task["target"] == [m["name"]]
        checks["no metric carries a role"] = "role" not in m
        checks["no metric carries an availability"] = "availability" not in m
        checks["the time block is gone from the core"] = (
            "time" not in prof and "x-deep" not in prof)

        # the suggested draft compiles conceptually: feed it through finalize
        prof_f, rep_f = self._finalize({"draft": prof})
        checks["suggested draft finalizes cleanly"] = rep_f["ok"] and rep_f["targets"] == 1
        checks["finalizing an already-split draft does not re-lift it"] = (
            rep_f["lifted_from_flat"] is False)
        checks["finalize is idempotent on its own output"] = (
            self._finalize({"draft": prof_f})[0] == prof_f)

        # ---- suggest: retail total sales ----
        retail = {
            "goal": "weekly total sales by store",
            "dataset": "pos",
            "schema": [
                {"name": "date", "type": "date"},
                {"name": "sales", "type": "number"},
                {"name": "store_nbr", "type": "string", "cardinality": 54},
                {"name": "family", "type": "string", "cardinality": 33},
            ],
        }
        prof2, _ = self._suggest(retail)
        checks["suggest picks sum for 'total'"] = prof2["metrics"][0]["function"] == "sum"
        checks["suggest infers week grain"] = prof2["task"]["clock"]["grain"] == "week"
        checks["suggest sum has no entity"] = "by" not in prof2["metrics"][0]

        # ---- triage: the goal picks the question ------------------------
        # Deterministic, ordered, and in FRONT of the language model: a total
        # LLM outage still produces a correctly-shaped task.
        triage_cases = {
            "forecast next quarter's revenue": "forecast",
            "quarterly supplier concentration by NAICS category": "forecast",
            "recommend items to customers": "recommend",
            "suggest movies for each viewer": "recommend",
            "rank the top 10 suppliers by spend": "rank",
            "flag unusual spikes in daily spend": "detect_anomaly",
            "find outliers in the transaction log": "detect_anomaly",
            "segment customers into cohorts": "cluster",
            "group accounts that behave alike": "cluster",
            "classify which customers churn": "classify",
            "predict whether an invoice will be paid late": "classify",
            "estimate a sale price for each home": "regress",
        }
        for goal, want in triage_cases.items():
            checks[f"triage: '{goal[:36]}' -> {want}"] = self.triage(goal.lower()) == want
        checks["triage reaches every catalogued kind"] = (
            set(triage_cases.values()) == set(task_library.kinds()))
        # The kind tables are derived, not hand-kept. That is the point: the
        # catalog says detect_anomaly only PERMITS a clock, so an outlier hunt
        # across entities is a legitimate question — which the hand-written
        # CLOCKED_KINDS tuple had wrong, because nothing forced it to agree.
        checks["the kind tables come from the one catalog"] = (
            set(CLOCKED_KINDS) == {"forecast"}
            and set(CLOCKED_KINDS) | set(CLOCKLESS_KINDS) == set(task_library.kinds()))
        checks["a clockless anomaly hunt is not refused"] = (
            self._suggest({"goal": "find outliers in customer spend",
                           "schema": [{"name": "cust_id", "type": "string",
                                       "cardinality": 40},
                                      {"name": "spend", "type": "number"}]})[0]
            ["task"]["kind"] == "detect_anomaly")

        # ---- the kinds the catalog had but nothing could produce ---------
        interactions = [
            {"name": "order_dt", "type": "date"},
            {"name": "cust_id", "type": "string", "cardinality": 500},
            {"name": "item_sku", "type": "string", "cardinality": 120},
            {"name": "stars", "type": "number"},
        ]
        clu, _ = self._suggest(
            {"goal": "segment customers into cohorts", "schema": interactions})
        checks["cluster: kind"] = clu["task"]["kind"] == "cluster"
        checks["cluster: keys on the entity the goal named"] = (
            clu["task"]["keys"] == ["cust_id"])
        checks["cluster: nothing is predicted"] = not any(
            f in clu["task"] for f in task_library.POINTERS)
        checks["cluster: every metric rides along as a covariate"] = (
            [c["metric"] for c in clu["task"]["covariates"]] == ["sum_stars"])
        checks["cluster: 'cohorts' does not invent a clock"] = "clock" not in clu["task"]
        checks["cluster finalizes, where it used to be rejected outright"] = (
            self._finalize({"draft": clu})[1]["ok"])

        # a permitted clock appears only when the goal asks for one
        clu_t, _ = self._suggest(
            {"goal": "segment customers into cohorts each month", "schema": interactions})
        checks["asking for time does add the clock back"] = (
            clu_t["task"]["clock"] == {"event": "order_dt", "grain": "month"})

        reg, _ = self._suggest({
            "goal": "estimate a sale price for each home",
            "schema": [{"name": "hood", "type": "string", "cardinality": 30},
                       {"name": "price", "type": "number"},
                       {"name": "beds", "type": "number"}]})
        checks["regress: kind"] = reg["task"]["kind"] == "regress"
        checks["regress: the goal's own word picks the target"] = (
            reg["task"]["target"] == ["sum_price"])
        checks["regress: the rest ride along"] = (
            [c["metric"] for c in reg["task"]["covariates"]] == ["sum_beds"])
        checks["regress: no clock unless asked"] = "clock" not in reg["task"]

        rnk, _ = self._suggest(
            {"goal": "rank the top 10 items for each customer", "schema": interactions})
        checks["rank: kind"] = rnk["task"]["kind"] == "rank"
        checks["rank: two keys and a signal, like recommend"] = (
            len(rnk["task"]["keys"]) == 2 and "signal" in rnk["task"])

        anom, _ = self._suggest(
            {"goal": "flag unusual spikes in monthly spend", "schema": interactions})
        checks["detect_anomaly: kind"] = anom["task"]["kind"] == "detect_anomaly"
        checks["detect_anomaly: scores a signal over a clock"] = (
            "signal" in anom["task"] and "clock" in anom["task"])

        rec, _ = self._suggest(
            {"goal": "recommend items to customers", "schema": interactions})
        checks["recommend: scores a signal, not a target"] = (
            "signal" in rec["task"] and "target" not in rec["task"])

        # ---- suggest edge: no time column DEMOTES, it does not error ----
        # This asserted a raise. Refusing here vetoes the language model: the
        # scaffold always runs first and the LLM stage corrects its draft, so a
        # raise halts the pipeline before the model reads the sentence at all.
        # The contract is now "always emit a draft, and say what you demoted".
        try:
            prof_nc, rep_nc = self._suggest({
                "goal": "predict who will survive",
                "schema": [{"name": "fare", "type": "number", "sample": "7.25",
                            "cardinality": 200},
                           {"name": "survived", "type": "number", "sample": "1",
                            "cardinality": 2},
                           {"name": "pid", "type": "string", "sample": "p1",
                            "cardinality": 300}]})
            checks["no clock demotes instead of raising"] = True
            checks["the demoted draft is clockless"] = (
                "clock" not in (prof_nc.get("task") or {}))
            checks["and it names the demotion"] = any(
                "drafted a" in w for w in rep_nc.get("warnings", []))
        except ValueError:
            checks["no clock demotes instead of raising"] = False
            checks["the demoted draft is clockless"] = False
            checks["and it names the demotion"] = False

        # ---- suggest fallback: hhi asked for but no entity -> sum ----
        prof3, rep3 = self._suggest({
            "goal": "monthly concentration",
            "schema": [{"name": "d", "type": "date"}, {"name": "amt", "type": "number"}]})
        checks["hhi without entity falls back to sum"] = (
            prof3["metrics"][0]["function"] == "sum" and any("entity" in w for w in rep3["warnings"]))

        # ---- finalize: normalize a hand draft ----
        draft = {
            "name": "sales", "version": "2",
            "metrics": [
                {"name": "revenue", "function": "sum", "measure": "amount", "role": "target"},
                {"name": "promo", "function": "max", "measure": "on_promo", "role": "feature"},
            ],
            "dimensions": [{"name": "store"}],
            "time": {"event": "day", "grain": "WEEK"},
            "x-deep": {"series": "store"},
        }
        fp, fr = self._finalize({"draft": draft})
        checks["finalize lowercases grain"] = fp["task"]["clock"]["grain"] == "week"
        checks["finalize defaults covariate availability"] = (
            fp["task"]["covariates"] == [{"metric": "promo", "availability": "past_only"}])
        checks["finalize warns on defaulted availability"] = any(
            "availability" in w for w in fr["warnings"])
        checks["finalize counts targets/features"] = fr["targets"] == 1 and fr["features"] == 1
        checks["finalize lifts a flat draft"] = fr["lifted_from_flat"] is True
        checks["finalize lifts role into the kind's pointer"] = (
            fp["task"]["target"] == ["revenue"])
        checks["finalize lifts x-deep.series into task.keys"] = (
            fp["task"]["keys"] == ["store"])
        checks["finalize strips role and availability off the metrics"] = not any(
            "role" in m or "availability" in m for m in fp["metrics"])
        checks["finalize moves time into the task"] = (
            "time" not in fp and "x-deep" not in fp)

        # a kind whose pointer is not `target` lifts a legacy role just the same
        legacy_rec = {
            "name": "recs", "kind": "recommend",
            "keys": [{"name": "k_cust", "from": "cust"},
                     {"name": "k_item", "from": "item"}],
            "dimensions": [{"name": "cust"}, {"name": "item"}],
            "metrics": [{"name": "rating", "function": "mean",
                         "measure": "rating", "role": "target"}]}
        lp, lr = self._finalize({"draft": legacy_rec})
        checks["a legacy role lifts to whatever pointer the kind uses"] = (
            lp["task"]["signal"] == ["rating"] and "target" not in lp["task"])
        checks["the report says which pointer was used"] = lr["pointer"] == "signal"
        checks["an explicit keys[] survives finalize"] = (
            lp["keys"] == legacy_rec["keys"] and lp["task"]["keys"] == ["cust", "item"])

        # ---- finalize: validation errors ----
        def fails(d, needle):
            try:
                self._finalize({"draft": d})
                return False
            except ValueError as e:
                return needle in str(e)

        checks["finalize rejects unknown function"] = fails(
            {"metrics": [{"name": "m", "function": "wizardry", "role": "target"}],
             "time": {"event": "d", "grain": "week"}}, "unknown function")
        checks["finalize rejects a forecast with nothing to predict"] = fails(
            {"metrics": [{"name": "m", "function": "sum", "measure": "x", "role": "feature"}],
             "time": {"event": "d", "grain": "week"}}, "requires 'target'")
        checks["finalize rejects group fn without by"] = fails(
            {"metrics": [{"name": "m", "function": "hhi", "measure": "x", "role": "target"}],
             "time": {"event": "d", "grain": "week"}}, "no 'by'")
        checks["finalize rejects bad grain"] = fails(
            {"metrics": [{"name": "m", "function": "sum", "measure": "x", "role": "target"}],
             "time": {"event": "d", "grain": "fortnight"}}, "grain")

        # ---- finalize now holds the draft against the catalog -------------
        # The last step before a human signs, so it fails here rather than one
        # service later with the same message.
        cluster_draft = {
            "name": "segments",
            "metrics": [{"name": "spend", "function": "sum", "measure": "amt"}],
            "dimensions": [{"name": "cust"}],
            "task": {"kind": "cluster", "keys": ["cust"],
                     "covariates": [{"metric": "spend"}]}}
        checks["finalize accepts a task that predicts nothing"] = (
            self._finalize({"draft": cluster_draft})[1]["predicted"] == [])
        checks["finalize rejects an unknown kind"] = fails(
            {**cluster_draft, "task": {**cluster_draft["task"], "kind": "telepathy"}},
            "kind")
        checks["finalize rejects a field the kind cannot use"] = fails(
            {**cluster_draft, "task": {**cluster_draft["task"], "target": "spend"}},
            "has no concept of 'target'")
        checks["finalize rejects a task field outside the vocabulary"] = fails(
            {**cluster_draft, "task": {**cluster_draft["task"], "horizon": 8}},
            "not part of the task vocabulary")
        checks["finalize rejects a pointer to an undeclared metric"] = fails(
            {**cluster_draft,
             "task": {"kind": "regress", "keys": ["cust"], "target": "nonesuch"}},
            "not declared")
        checks["finalize rejects a key that is not a dimension"] = fails(
            {**cluster_draft, "task": {**cluster_draft["task"], "keys": ["nowhere"]}},
            "not a declared dimension")
        checks["finalize rejects a metric that is both predicted and carried"] = fails(
            {**cluster_draft,
             "task": {"kind": "regress", "keys": ["cust"], "target": "spend",
                      "covariates": [{"metric": "spend"}]}},
            "both predicted and a covariate")
        checks["finalize rejects too few keys for the kind"] = fails(
            {**cluster_draft,
             "task": {"kind": "recommend", "keys": ["cust"], "signal": "spend"}},
            "needs 2..2 key(s)")
        checks["finalize still demands a clock where the kind needs one"] = fails(
            {"metrics": [{"name": "m", "function": "sum", "measure": "x", "role": "target"}],
             "dimensions": [{"name": "d"}]}, "required for kind 'forecast'")
        checks["the old kind name still resolves"] = (
            self._finalize({"draft": {
                "kind": "anomaly", "dimensions": [{"name": "d"}],
                "metrics": [{"name": "m", "function": "sum", "measure": "x",
                             "role": "target"}],
                "time": {"event": "d", "grain": "week"}}})[1]["kind"]
            == "detect_anomaly")

        # the catalogs must match their embedded fallbacks
        checks["function csv matches embedded fallback"] = (
            function_library.catalog() == function_library.embedded_catalog())
        checks["task csv matches embedded fallback"] = (
            task_library.catalog() == task_library.embedded_catalog())

        passed = sum(1 for v in checks.values() if v)
        return {
            "service": "profile-generator",
            "passed": passed,
            "total": len(checks),
            "ok": passed == len(checks),
            "checks": checks,
        }
