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

from spl.core.base_service.base_service_class import BaseService
from spl.core.service_types import ParameterEnum

try:  # sibling module: package-relative when deployed, flat when run locally
    from . import function_library
except ImportError:  # pragma: no cover
    import function_library

GRAINS = ("day", "week", "month", "quarter", "year")
ROLES = ("target", "feature")
AVAIL = ("known_ahead", "past_only")

# name-shape hints used only by the suggest heuristics
_DATEISH = re.compile(r"(date|_dt$|^dt$|time|period|timestamp|^ds$|_at$|day)", re.I)
_ENTITYISH = re.compile(r"(vendor|recipient|parent|supplier|customer|company|"
                        r"account|entity|name|_id$|^id$)", re.I)
_DIMISH = re.compile(r"(category|type|class|region|state|store|segment|group|"
                     r"naics|sector|dept|department|channel|product|market|branch)", re.I)
_NUMTYPES = ("number", "numeric", "integer", "int", "float", "double", "decimal", "money")
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

    def _suggest(self, p):
        cols = self._read_schema(p.get("schema"))
        if not cols:
            raise ValueError("suggest needs a schema (a non-empty list of columns)")
        goal = str(p.get("goal") or "").strip()
        goal_l = goal.lower()
        dataset = str(p.get("dataset") or "dataset").strip() or "dataset"
        warnings = []

        # -- time column (no forecasting without one) --------------------
        date_col = self._pick(cols, self._is_date)
        if not date_col:
            raise ValueError(
                "no time column found in the schema (need a date/timestamp to forecast)")

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

        time = draft.get("time")
        if not isinstance(time, dict) or not str(time.get("event") or "").strip():
            raise ValueError("draft.time.event is required")
        grain = str(time.get("grain") or "").strip().lower()
        if grain not in GRAINS:
            raise ValueError(f"draft.time.grain must be one of {GRAINS}, got '{time.get('grain')}'")

        dim_names = []
        for d in (draft.get("dimensions") or []):
            if isinstance(d, dict) and str(d.get("name") or "").strip():
                nm = str(d["name"]).strip()
                if nm not in dim_names:
                    dim_names.append(nm)

        norm_metrics = []
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
            role = str(m.get("role") or "").strip().lower()
            if role not in ROLES:
                raise ValueError(f"metric '{name}' needs role one of {ROLES}")

            nm = {"name": name, "function": fn, "role": role}
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

            if role == "target":
                n_targets += 1
            else:
                avail = str(m.get("availability") or "").strip().lower()
                if avail and avail not in AVAIL:
                    raise ValueError(
                        f"feature '{name}' availability must be one of {AVAIL}, got '{avail}'")
                if not avail:
                    avail = "past_only"
                    warnings.append(f"feature '{name}' had no availability; "
                                    f"defaulted to past_only")
                nm["availability"] = avail
            norm_metrics.append(nm)

        if n_targets == 0:
            raise ValueError("no metric has role 'target' - nothing to forecast")

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
            "time": {"event": str(time["event"]).strip(), "grain": grain},
        }
        if series:
            profile["x-deep"] = {"series": series}

        report = {
            "mode": "finalize",
            "ok": True,
            "metrics": len(norm_metrics),
            "targets": n_targets,
            "features": len(norm_metrics) - n_targets,
            "warnings": warnings,
            "note": ("Validated + normalized. Safe to sign and hand to spec-compiler "
                     "(which will still re-check against the binding)."),
        }
        return profile, report

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

    def _is_date(self, c):
        if c["type"] in _DATETYPES:
            return True
        if c["type"] in _NUMTYPES:
            return False
        return bool(_DATEISH.search(c["name"]))

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
        m = prof["metrics"][0]
        checks["suggest picks hhi from 'concentration'"] = m["function"] == "hhi"
        checks["suggest infers quarter grain"] = prof["time"]["grain"] == "quarter"
        checks["suggest binds time to the date col"] = prof["time"]["event"] == "action_dt"
        checks["suggest picks the numeric measure"] = m["measure"] == "dollars_obligated"
        checks["suggest picks an entity for hhi"] = m["by"] == "recipient_parent"
        checks["suggest picks a series dimension"] = prof.get("x-deep", {}).get("series") == "naics_code"
        checks["suggest emits an identity binding stub"] = (
            rep["binding_stub"]["bind"].get("action_dt") == "action_dt")
        checks["suggest target role"] = m["role"] == "target"

        # the suggested draft compiles conceptually: feed it through finalize
        prof_f, rep_f = self._finalize({"draft": prof})
        checks["suggested draft finalizes cleanly"] = rep_f["ok"] and rep_f["targets"] == 1

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
        checks["suggest infers week grain"] = prof2["time"]["grain"] == "week"
        checks["suggest sum has no entity"] = "by" not in prof2["metrics"][0]

        # ---- suggest edge: no time column is an error ----
        try:
            self._suggest({"goal": "sales", "schema": [{"name": "sales", "type": "number"}]})
            checks["suggest without a time column errors"] = False
        except ValueError as e:
            checks["suggest without a time column errors"] = "time column" in str(e)

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
        checks["finalize lowercases grain"] = fp["time"]["grain"] == "week"
        checks["finalize defaults feature availability"] = (
            fp["metrics"][1]["availability"] == "past_only")
        checks["finalize warns on defaulted availability"] = any(
            "availability" in w for w in fr["warnings"])
        checks["finalize counts targets/features"] = fr["targets"] == 1 and fr["features"] == 1

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
        checks["finalize rejects no target"] = fails(
            {"metrics": [{"name": "m", "function": "sum", "measure": "x", "role": "feature"}],
             "time": {"event": "d", "grain": "week"}}, "nothing to forecast")
        checks["finalize rejects group fn without by"] = fails(
            {"metrics": [{"name": "m", "function": "hhi", "measure": "x", "role": "target"}],
             "time": {"event": "d", "grain": "week"}}, "no 'by'")
        checks["finalize rejects bad grain"] = fails(
            {"metrics": [{"name": "m", "function": "sum", "measure": "x", "role": "target"}],
             "time": {"event": "d", "grain": "fortnight"}}, "grain")

        passed = sum(1 for v in checks.values() if v)
        return {
            "service": "profile-generator",
            "passed": passed,
            "total": len(checks),
            "ok": passed == len(checks),
            "checks": checks,
        }
