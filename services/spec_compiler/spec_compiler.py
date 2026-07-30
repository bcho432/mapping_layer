"""spec-compiler: turn an OSI-compatible profile + binding into a mapping-layer SPEC.

This is the deterministic middle of the Decision Engine's three-layer stack:

    profile (what things MEAN, portable)          <- authored/signed upstream
        + binding (where each meaning LIVES)       <- set per data source
    ------------------------------------------------------------------
    spec-compiler  (this service, deterministic)
    ------------------------------------------------------------------
        -> mapping-layer SPEC (physical columns)   -> the mapping layer executes

The profile is vendor-neutral (OSI shape: datasets / metrics / dimensions /
time, plus an `x-deep` forecasting extension). It never names a physical
column. The binding maps each logical name to a real column in one customer's
data. The compiler joins the two and lowers them to the SPEC the mapping layer
speaks -- so the LLM/human authoring loop stays away from physical columns, and
the deterministic lowering stays away from the LLM.

HOW IT LOWERS. Stage 4 does not fill in a template. It walks the profile and
each element raises *obligations*: "this path in the SPEC must hold this value,
and here is where it came from". Obligations are resolved against the binding
and written to a pad that refuses conflicting writes, then collected into the
document and checked against the contract for the question's `kind`. Adding a
slot means adding an emitter; adding a function argument means adding a catalog
row. Nothing here has an opinion about business vocabulary.

The emitted SPEC is positioned: keys (what a row is), aggregates (the numbers),
derives (calendar columns), filters (which rows count) and validity (when a row
became knowable). Time is not a special slot -- it is a key with a bin: derive.

Input   request.parameters.profile   the OSI profile (parsed object)
        request.parameters.binding   the physical binding (parsed object)
        request.parameters.naming    'v1' (default) keeps the frame's legacy
                                     column names (series_id / t / y / x_*);
                                     'logical' names them after the profile.
                                     This is a dialect choice and moves to the
                                     engine adapter once the run config lands.
Output  RunResponse.data     [ <the mapping-layer SPEC> ]  (one row)
        RunResponse.metadata compile report: the stages, a field-by-field
                             crosswalk generated from the walk, the functions
                             used, and any warnings

Deterministic and stdlib only - no LLM, no package_dependencies. Operates on
parsed objects, not raw YAML (parse upstream). Ships its own copy of
function_library and derive_library so it can validate names without the
mapping layer.
"""

import asyncio
import json
from dataclasses import dataclass

from spl.core.base_service.base_service_class import BaseService
from spl.core.service_types import ParameterEnum

try:  # sibling modules: package-relative when deployed, flat when run locally
    from . import derive_library, function_library
except ImportError:  # pragma: no cover
    import derive_library
    import function_library

GRAINS = derive_library.GRAINS
ROLES = ("target", "feature")
AVAIL = ("known_ahead", "past_only")
NAMINGS = ("v1", "logical")

# Which SPEC field a catalogued `needs` token lowers into. A catalog row that
# names a need with no entry here is a gap the compiler reports rather than
# silently drops -- the same three-state contract the libraries use.
NEED_SLOT = {"measure": "of", "entity": "by"}

# How to read that need's argument off a profile metric, and what to call it
# when it is missing.
NEED_ARG = {
    "measure": lambda m: str(m.get("measure") or "").strip(),
    "entity": lambda m: str(m.get("by") or m.get("entity") or "").strip(),
}
NEED_MSG = {"measure": "a measure", "entity": "a 'by' entity"}

# The contract, as data. R require / P permit / F forbid, and a (min, max) count
# of entity keys -- the keys that are not a time bin. This is the only place the
# kinds exist: the mapping layer never learns what a `kind` is.
CONTRACT = {
    "forecast":  {"keys": (0, 1), "time_from": "R", "time_bin": "R",
                  "targets": "R", "features": "P", "validity": "P"},
    "anomaly":   {"keys": (0, 1), "time_from": "R", "time_bin": "R",
                  "targets": "R", "features": "P", "validity": "P"},
    "classify":  {"keys": (1, 2), "time_from": "P", "time_bin": "P",
                  "targets": "R", "features": "R", "validity": "F"},
    "cluster":   {"keys": (1, 2), "time_from": "P", "time_bin": "P",
                  "targets": "F", "features": "R", "validity": "F"},
    "regress":   {"keys": (1, 2), "time_from": "F", "time_bin": "F",
                  "targets": "R", "features": "R", "validity": "F"},
    "recommend": {"keys": (2, 2), "time_from": "F", "time_bin": "F",
                  "targets": "R", "features": "P", "validity": "F"},
    "rank":      {"keys": (2, 2), "time_from": "F", "time_bin": "F",
                  "targets": "R", "features": "R", "validity": "F"},
}
KINDS = tuple(CONTRACT)


# --------------------------------------------------------------- the machinery

@dataclass(frozen=True)
class Obligation:
    """One value the SPEC must carry, and where it came from.

    Exactly one of `logical` (resolve it through the binding) or `literal`
    (copy it straight through) is set. That distinction is the whole difference
    between a name the customer's data has to supply and a choice the profile
    made, and keeping it explicit is what lets one loop both resolve and
    build the crosswalk.
    """
    path: str
    logical: str = None
    literal: object = None
    origin: str = ""
    source: str = None

    def __post_init__(self):
        if (self.logical is None) == (self.literal is None):
            raise ValueError(
                f"obligation {self.path}: set exactly one of logical/literal")


class Pad:
    """Write-once accumulator keyed by SPEC path.

    Writing the same value twice is a no-op, so emitters stay independent.
    Writing a different value is an error naming both origins -- the failure a
    plain dict.update() hides, and the one that silently produces a wrong frame
    once more than one source can write the same slot.
    """

    def __init__(self):
        self.slots = {}

    def write(self, path, value, origin):
        prev = self.slots.get(path)
        if prev is not None:
            if prev[0] != value:
                raise ValueError(
                    f"conflict at {path}: {prev[1]} wrote {prev[0]!r}, "
                    f"{origin} wrote {value!r}")
            return          # idempotent: keep the origin that established it
        self.slots[path] = (value, origin)

    def origin_of(self, path):
        hit = self.slots.get(path)
        return hit[1] if hit else "(nothing)"


def _unflatten(slots):
    """Flat path map -> the nested SPEC.

    List indices are renumbered densely in first-write order: an emitter may
    number `aggregates[j]` from a metric's position in the profile, and targets
    and features share that counter, so the raw indices can have holes.
    """
    out = {}
    dense = {}
    for path, (value, _origin) in slots.items():
        node, prefix = out, ""
        parts = path.split(".")
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            if part.endswith("]") and "[" in part:
                nm, raw_idx = part[:-1].split("[", 1)
                slot = f"{prefix}|{nm}[{raw_idx}]"
                lst = node.setdefault(nm, [])
                if slot not in dense:
                    dense[slot] = len(lst)
                    lst.append({})
                if last:
                    raise ValueError(f"path '{path}' ends at a list element")
                node = lst[dense[slot]]
            elif last:
                node[part] = value
            else:
                node = node.setdefault(part, {})
            prefix = f"{prefix}/{part}"
    return out


class SpecCompilerService(BaseService):

    # ------------------------------------------------------------------ run

    async def _run(self, request: BaseService.RunRequest) -> BaseService.RunResponse:
        p = self._param_root(request.parameters or {})
        try:
            profile = self._as_object(p.get("profile"), "profile")
            binding = self._as_object(p.get("binding"), "binding")
            naming = str(p.get("naming") or "v1").strip().lower()
            if naming not in NAMINGS:
                raise ValueError(f"naming must be one of {NAMINGS}, got '{naming}'")
            spec, report = await asyncio.to_thread(
                self._compile, profile, binding, naming)
        except ValueError as e:
            self.logger.error(f"spec-compiler: {e}")
            return BaseService.RunResponse(data=[{"error": str(e)}])

        if report["warnings"]:
            self.logger.warning(
                f"spec-compiler: compiled with {len(report['warnings'])} warning(s): "
                + "; ".join(report["warnings"]))
        roles = [a.get("role") for a in spec.get("aggregates", [])]
        self.logger.info(
            f"spec-compiler: profile '{report['profile']}' kind '{report['kind']}' -> SPEC "
            f"({roles.count('target')} target(s), {roles.count('feature')} feature(s), "
            f"{len(spec.get('keys', []))} key(s))")
        return BaseService.RunResponse(data=[spec], metadata=report)

    # --------------------------------------------------------------- inputs

    @staticmethod
    def _param_root(parameters):
        inner = parameters.get("serviceInstructions")
        return inner if isinstance(inner, dict) else parameters

    @staticmethod
    def _as_object(val, what):
        """Accept a parsed object, or a JSON string (YAML must be parsed upstream)."""
        if isinstance(val, dict):
            return val
        if isinstance(val, str) and val.strip():
            try:
                obj = json.loads(val)
            except ValueError:
                raise ValueError(
                    f"{what} must be an object (got a string that isn't JSON - "
                    f"parse YAML to an object upstream)")
            if not isinstance(obj, dict):
                raise ValueError(f"{what} must be an object, not {type(obj).__name__}")
            return obj
        raise ValueError(f"{what} is required (an object with the {what} document)")

    # ------------------------------------------------------------- the walk

    @staticmethod
    def _keys_of(profile, naming):
        """The profile's keys, as a list, in a fixed order.

        `profile.keys` is the general form. The named blocks are sugar over it:
        an x-deep series becomes one key, and a time grain becomes a bin: key.
        Nothing downstream can tell which form was authored.
        """
        raw = profile.get("keys")
        if isinstance(raw, list) and raw:
            out = []
            for i, k in enumerate(raw):
                if not isinstance(k, dict):
                    raise ValueError(f"profile.keys[{i}] must be an object")
                name = str(k.get("name") or "").strip()
                frm = str(k.get("from") or "").strip()
                via = str(k.get("via") or "").strip()
                if not name:
                    raise ValueError(f"profile.keys[{i}].name is required")
                if via:
                    derive_library.resolve(via, f"profile.keys[{i}].via")
                if not frm and not via.startswith("bin:"):
                    raise ValueError(
                        f"profile.keys[{i}] needs a 'from' (only a bin: key may omit it)")
                out.append({"name": name, "from": frm, "via": via,
                            "origin": f"profile.keys[{i}] = {name}"})
            return out

        out = []
        xdeep = profile.get("x-deep") or profile.get("x_deep") or {}
        if not isinstance(xdeep, dict):
            raise ValueError("profile.x-deep must be an object when present")
        series = str(xdeep.get("series") or "").strip()
        if series:
            out.append({"name": "series_id" if naming == "v1" else series,
                        "from": series, "via": "",
                        "origin": f"profile.x-deep.series = {series}"})
        time = profile.get("time") or {}
        grain = str(time.get("grain") or "").strip().lower()
        if grain:
            out.append({"name": "t" if naming == "v1" else "period",
                        "from": "", "via": "bin:" + grain,
                        "origin": f"profile.time.grain = {grain}"})
        return out

    def _walk(self, profile, binding, lib, naming):
        """Yield every obligation the profile raises, in a fixed order.

        Lists in list order, catalog `needs` in catalog order. Nothing here
        iterates a set: two runs of the same profile must produce byte-identical
        JSON, not merely equal dicts.
        """
        # -- P3 the clock column (the keys carry the grain) ------------------
        time = profile.get("time") or {}
        if not isinstance(time, dict):
            raise ValueError("profile.time must be an object when present")
        event = str(time.get("event") or "").strip()
        if event:
            yield Obligation("time_from", logical=event, origin="profile.time.event")

        # -- P5 keys ---------------------------------------------------------
        for i, k in enumerate(self._keys_of(profile, naming)):
            yield Obligation(f"keys[{i}].name", literal=k["name"], origin=k["origin"])
            if k["from"]:
                yield Obligation(f"keys[{i}].from", logical=k["from"], origin=k["origin"])
            if k["via"]:
                yield Obligation(f"keys[{i}].via", literal=k["via"], origin=k["origin"])

        # -- P6 aggregates ---------------------------------------------------
        metrics = profile.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise ValueError("profile.metrics must be a non-empty list")
        targets = [m for m in metrics if isinstance(m, dict)
                   and str(m.get("role") or "").strip().lower() == "target"]
        single_target = len(targets) == 1
        for j, m in enumerate(metrics):
            if not isinstance(m, dict):
                raise ValueError("each metric must be an object")
            mname = str(m.get("name") or "").strip()
            if not mname:
                raise ValueError("each metric needs a name")
            role = str(m.get("role") or "").strip().lower()
            if role not in ROLES:
                raise ValueError(
                    f"metric '{mname}' needs role one of {ROLES}, got '{m.get('role')}'")
            fn = str(m.get("function") or "").strip()
            if not fn:
                raise ValueError(f"metric '{mname}' has no function")
            entry = lib.get(fn)
            if entry is None:
                raise ValueError(
                    f"metric '{mname}' uses unknown function '{fn}' "
                    f"(not in the function library: {sorted(lib)})")

            if naming == "v1":
                out = ("y" if (role == "target" and single_target)
                       else mname if role == "target" else "x_" + mname)
            else:
                out = mname
            base = f"aggregates[{j}]"
            yield Obligation(f"{base}.name", literal=out, origin=f"metric {mname}")
            yield Obligation(f"{base}.using", literal=fn,
                             origin=f"metric {mname}.function")
            yield Obligation(f"{base}.role", literal=role, origin=f"metric {mname}.role")

            needs = entry.get("needs", [])
            for need in needs:
                slot = NEED_SLOT.get(need)
                if slot is None:
                    raise ValueError(
                        f"metric '{mname}' function '{fn}' declares need '{need}', which "
                        f"the compiler cannot lower (known: {sorted(NEED_SLOT)}). Add it to "
                        f"NEED_SLOT/NEED_ARG in spec_compiler.py")
                arg = NEED_ARG[need](m)
                if not arg:
                    raise ValueError(
                        f"metric '{mname}' uses function '{fn}' which needs "
                        f"{NEED_MSG.get(need, need)}")
                yield Obligation(f"{base}.{slot}", logical=arg,
                                 origin=f"metric {mname}.{need}")
            # a measure supplied to a function that does not require one is
            # still honoured (count over a specific column, for instance)
            spare = NEED_ARG["measure"](m)
            if spare and "measure" not in needs:
                yield Obligation(f"{base}.of", logical=spare,
                                 origin=f"metric {mname}.measure")

            if role == "feature":
                avail = str(m.get("availability") or "").strip().lower() or "past_only"
                if avail not in AVAIL:
                    raise ValueError(
                        f"feature '{mname}' availability must be one of {AVAIL}, got '{avail}'")
                yield Obligation(f"{base}.availability", literal=avail,
                                 origin=f"metric {mname}.availability")

        # -- P2 filters ------------------------------------------------------
        for n, f in enumerate(profile.get("filters") or []):
            if not isinstance(f, dict):
                raise ValueError(f"profile.filters[{n}] must be an object")
            col = str(f.get("of") or f.get("column") or "").strip()
            if not col:
                raise ValueError(f"profile.filters[{n}] needs a column ('of')")
            op = str(f.get("using") or f.get("operator") or "equals").strip()
            base = f"filters[{n}]"
            yield Obligation(f"{base}.of", logical=col, origin=f"profile.filters[{n}]")
            yield Obligation(f"{base}.using", literal=op, origin=f"profile.filters[{n}]")
            if f.get("value") is not None:
                yield Obligation(f"{base}.value", literal=f.get("value"),
                                 origin=f"profile.filters[{n}]")

        # -- P4 validity -----------------------------------------------------
        # A logical arrival name on the profile is resolved like any other; the
        # binding's available_from is already a physical column, so it copies.
        arrival_logical = str(time.get("arrival") or "").strip()
        arrival_physical = str(binding.get("available_from") or "").strip()
        if arrival_logical:
            yield Obligation("validity.arrival_from", logical=arrival_logical,
                             origin="profile.time.arrival")
        elif arrival_physical:
            yield Obligation("validity.arrival_from", literal=arrival_physical,
                             origin="binding.available_from")

    # ------------------------------------------------------------ contract

    @staticmethod
    def _check(spec, kind, pad):
        """Hold the emitted SPEC against the contract for this kind.

        A template fails loudly -- a slot you forgot to fill is a visible empty
        box. A walk fails by omission, which is quiet. This is what closes that
        gap, and it is the reason the walk is allowed to be permissive.
        """
        rules = CONTRACT.get(kind)
        if rules is None:
            raise ValueError(f"unknown kind '{kind}' (known: {sorted(CONTRACT)})")

        keys = spec.get("keys") or []
        aggs = spec.get("aggregates") or []
        entity_keys = [k for k in keys if not str(k.get("via") or "").startswith("bin:")]
        facts = {
            "time_from": bool(spec.get("time_from")),
            "time_bin": any(str(k.get("via") or "").startswith("bin:") for k in keys),
            "targets": any(a.get("role") == "target" for a in aggs),
            "features": any(a.get("role") == "feature" for a in aggs),
            "validity": bool((spec.get("validity") or {}).get("arrival_from")),
        }
        blame = {"time_from": "time_from", "time_bin": "keys",
                 "targets": "aggregates", "features": "aggregates",
                 "validity": "validity.arrival_from"}
        for slot, rule in rules.items():
            if slot == "keys":
                continue
            if rule == "R" and not facts[slot]:
                raise ValueError(
                    f"kind '{kind}' requires '{slot}', but nothing emitted it")
            if rule == "F" and facts[slot]:
                raise ValueError(
                    f"kind '{kind}' forbids '{slot}', but "
                    f"{pad.origin_of(blame[slot])} emitted it")
        lo, hi = rules["keys"]
        if not lo <= len(entity_keys) <= hi:
            raise ValueError(
                f"kind '{kind}' needs {lo}..{hi} entity key(s), got {len(entity_keys)} "
                f"({[k.get('name') for k in entity_keys]})")

    # ------------------------------------------------------------- compile

    def _compile(self, profile, binding, naming="v1"):
        """The five deterministic stages. Returns (spec, report)."""
        report = {
            "profile": "",
            "kind": "",
            "source": "",
            "naming": naming,
            "stages": [],
            "crosswalk": [],       # [{profile, resolver, spec}]
            "functions_used": [],
            "warnings": [],
        }

        # -- stage 1: parse + validate the profile -----------------------
        name = str(profile.get("name") or "").strip()
        version = str(profile.get("version") or "").strip()
        report["profile"] = f"{name} v{version}".strip() if name else "(unnamed profile)"

        kind = str(profile.get("kind") or "forecast").strip().lower()
        if kind not in CONTRACT:
            raise ValueError(f"profile.kind must be one of {sorted(CONTRACT)}, got '{kind}'")
        report["kind"] = kind

        time = profile.get("time") or {}
        if not isinstance(time, dict):
            raise ValueError("profile.time must be an object when present")
        grain = str(time.get("grain") or "").strip().lower()
        if grain and grain not in GRAINS:
            raise ValueError(f"profile.time.grain must be one of {GRAINS}, got '{grain}'")
        if grain and not str(time.get("event") or "").strip():
            raise ValueError(
                "profile.time.grain is set but profile.time.event is missing "
                "(which logical name carries the timestamp?)")

        dim_names = [str(d.get("name") or "").strip()
                     for d in (profile.get("dimensions") or []) if isinstance(d, dict)]
        for k in self._keys_of(profile, naming):
            if k["from"] and dim_names and k["from"] not in dim_names:
                report["warnings"].append(
                    f"key source '{k['from']}' is not listed under dimensions")
        report["stages"].append("parse+validate: profile is structurally valid")

        # -- stage 2: resolve the binding --------------------------------
        bind = binding.get("bind")
        if not isinstance(bind, dict):
            raise ValueError("binding.bind must be an object mapping logical names to columns")
        report["source"] = str(binding.get("source") or "").strip()

        def resolve(logical, where):
            col = str(bind.get(logical) or "").strip()
            if not col:
                raise ValueError(
                    f"unbound: '{logical}' (needed for {where}) has no column in binding.bind")
            return col

        report["stages"].append(f"resolve binding: source '{report['source']}'")

        # -- stage 3: the function library -------------------------------
        lib = function_library.catalog()

        # -- stage 4: walk the profile, emit obligations, collect --------
        pad = Pad()
        for ob in self._walk(profile, binding, lib, naming):
            if ob.logical is not None:
                value = resolve(ob.logical, ob.origin)
                resolver = "binding"
            else:
                value = ob.literal
                resolver = "copy"
            pad.write(ob.path, value, ob.origin)
            report["crosswalk"].append(
                {"profile": ob.origin, "resolver": resolver,
                 "spec": f"{ob.path} = {value}"})
            if ob.path.endswith(".using") and value not in report["functions_used"]:
                if value in lib:
                    report["functions_used"].append(value)

        spec = _unflatten(pad.slots)

        # a group function given a 'by' it cannot use is a warning, not an error
        for m in profile.get("metrics") or []:
            if not isinstance(m, dict):
                continue
            fn = str(m.get("function") or "").strip()
            entry = lib.get(fn) or {}
            if NEED_ARG["entity"](m) and "entity" not in entry.get("needs", []):
                report["warnings"].append(
                    f"metric '{m.get('name')}' has 'by: {NEED_ARG['entity'](m)}' but "
                    f"'{fn}' is not a group function; ignoring")

        report["stages"].append(
            "resolve functions: " + (", ".join(report["functions_used"]) or "(none)"))
        report["stages"].append(
            f"walk+emit: {len(pad.slots)} obligation(s) over "
            f"{len(spec.get('keys', []))} key(s), {len(spec.get('aggregates', []))} aggregate(s)")

        # -- stage 5: check the contract ---------------------------------
        self._check(spec, kind, pad)
        report["stages"].append(f"check: SPEC satisfies the '{kind}' contract")
        return spec, report

    # -------------------------------------------------------------- schema

    async def schema(self) -> dict | None:
        SSP = BaseService.ServiceSchemaProperty
        dlib = derive_library.catalog()
        key_derives = [n for n, e in dlib.items() if e["scope"] == "key"]
        return SSP(
            key="serviceInstructions",
            type="object",
            description=("Compile an OSI-compatible profile + a physical binding into a "
                         "mapping-layer SPEC. The profile carries meaning (portable); the "
                         "binding carries the physical columns for one data source."),
            properties=[
                SSP(key="profile", type="object",
                    description=("The OSI profile (meaning). Usually produced by profile-generator "
                                 "and signed upstream; passed here as an object."),
                    properties=[
                        SSP(key="name", type="string",
                            description="Profile name (for the compile report)"),
                        SSP(key="version", type="string",
                            description="Profile version (for the compile report)"),
                        SSP(key="kind", type="enum", default="forecast",
                            description=("What kind of question this is. Decides which SPEC "
                                         "slots are required, permitted or forbidden"),
                            enum=ParameterEnum(values=list(KINDS))),
                        SSP(key="metrics", type="array",
                            description="What to compute; each becomes a target or a feature",
                            properties=[
                                SSP(key="name", type="string", required=True,
                                    description="Logical metric name"),
                                SSP(key="function", type="string", required=True,
                                    description="Function from the library (e.g. sum, hhi)"),
                                SSP(key="measure", type="string",
                                    description="Logical measure the function reads"),
                                SSP(key="by", type="string",
                                    description="Logical entity for share grouping (group functions like hhi)"),
                                SSP(key="role", type="enum", required=True,
                                    description="target = predict it; feature = carry it alongside",
                                    enum=ParameterEnum(values=list(ROLES))),
                                SSP(key="availability", type="enum",
                                    description="Features only: is it knowable at prediction time?",
                                    enum=ParameterEnum(values=list(AVAIL))),
                            ]),
                        SSP(key="keys", type="array",
                            description=("What identifies one frame row. The general form; "
                                         "leave empty to derive it from x-deep.series and time"),
                            properties=[
                                SSP(key="name", type="string", required=True,
                                    description="Output column name for this key"),
                                SSP(key="from", type="string",
                                    description="Logical name to group on (omit only on a bin: key)"),
                                SSP(key="via", type="enum",
                                    description="Derive applied before grouping (e.g. bin:quarter)",
                                    enum=ParameterEnum(values=key_derives)),
                            ]),
                        SSP(key="filters", type="array",
                            description="Keep only rows that pass every condition",
                            properties=[
                                SSP(key="of", type="string", required=True,
                                    description="Logical name of the column to test"),
                                SSP(key="using", type="string", default="equals",
                                    description="Comparison operator"),
                                SSP(key="value", type="string",
                                    description="Comparison value"),
                            ]),
                        SSP(key="dimensions", type="array",
                            description="Dimensions (key grains and share entities live here)",
                            properties=[SSP(key="name", type="string", required=True)]),
                        SSP(key="time", type="object",
                            description=("Event time and grain. Omit entirely for a "
                                         "cross-sectional question with no clock"),
                            properties=[
                                SSP(key="event", type="string",
                                    description="Logical event-time name"),
                                SSP(key="grain", type="enum",
                                    description="Bucket size; lowered to a bin: key",
                                    enum=ParameterEnum(values=list(GRAINS))),
                                SSP(key="arrival", type="string",
                                    description=("Logical name of the arrival time, if the "
                                                 "profile rather than the binding carries it")),
                            ]),
                        SSP(key="x-deep", type="object",
                            description="Forecasting extension (namespaced); sugar over keys[]",
                            properties=[
                                SSP(key="series", type="string",
                                    description="Which dimension is the forecast series grain"),
                            ]),
                    ]),
                SSP(key="binding", type="object",
                    description="Where each logical name lives in one customer's data",
                    properties=[
                        SSP(key="source", type="string",
                            description="Data source id (for the report)"),
                        SSP(key="bind", type="object",
                            description="Map of logical name -> physical column"),
                        SSP(key="available_from", type="string",
                            description=("Optional physical arrival-time column, passed to the "
                                         "mapping layer's leakage guard")),
                    ]),
                SSP(key="naming", type="enum", default="v1",
                    description=("Which names the frame's columns get. 'v1' keeps the legacy "
                                 "series_id / t / y / x_* so existing consumers keep working; "
                                 "'logical' names them after the profile. A dialect choice - it "
                                 "moves to the engine adapter once the run config lands"),
                    enum=ParameterEnum(values=list(NAMINGS))),
            ],
        )

    # ------------------------------------------------------------ self-test

    def self_test(self):
        """Offline check: the supplier-concentration HHI example from the design doc,
        the obligation machinery, and one profile per row of the contract."""
        checks = {}

        # ---- the design doc's worked example: HHI supplier concentration ----
        profile = {
            "name": "supplier_concentration", "version": "3",
            "datasets": [{"name": "awards"}],
            "metrics": [
                {"name": "contract_concentration", "function": "hhi",
                 "measure": "obligation", "by": "vendor", "role": "target"},
            ],
            "dimensions": [{"name": "category"}, {"name": "vendor"}],
            "time": {"event": "action_date", "grain": "quarter"},
            "x-deep": {"series": "category"},
        }
        binding = {
            "source": "usaspending_awards",
            "bind": {
                "action_date": "action_dt",
                "obligation": "dollars_obligated",
                "vendor": "recipient_parent",
                "category": "naics_code",
            },
        }
        spec, report = self._compile(profile, binding)

        checks["event lowered to time_from"] = spec["time_from"] == "action_dt"
        checks["series lowered to a key"] = (
            spec["keys"][0] == {"name": "series_id", "from": "naics_code"})
        checks["grain lowered to a bin key"] = (
            spec["keys"][1] == {"name": "t", "via": "bin:quarter"})
        checks["single aggregate, one entry"] = len(spec["aggregates"]) == 1
        agg = spec["aggregates"][0]
        checks["aggregate fn is hhi"] = agg["using"] == "hhi"
        checks["aggregate measure bound"] = agg["of"] == "dollars_obligated"
        checks["group fn entity bound"] = agg["by"] == "recipient_parent"
        checks["single target named y under v1"] = agg["name"] == "y"
        checks["no validity block"] = "validity" not in spec
        checks["no filters block"] = "filters" not in spec
        checks["hhi recorded as used"] = report["functions_used"] == ["hhi"]
        checks["report names the source"] = report["source"] == "usaspending_awards"
        checks["report names the kind"] = report["kind"] == "forecast"
        checks["five stages recorded"] = len(report["stages"]) == 5
        checks["crosswalk traces the target"] = any(
            c["resolver"] == "binding" and "recipient_parent" in c["spec"]
            for c in report["crosswalk"])

        expected = {
            "time_from": "action_dt",
            "keys": [{"name": "series_id", "from": "naics_code"},
                     {"name": "t", "via": "bin:quarter"}],
            "aggregates": [{"name": "y", "using": "hhi", "role": "target",
                            "of": "dollars_obligated", "by": "recipient_parent"}],
        }
        checks["compiled SPEC matches the design doc"] = spec == expected

        # the crosswalk is generated, so it covers the document by construction
        emitted = set()
        for path in ("time_from", "keys[0].name", "keys[0].from", "keys[1].name",
                     "keys[1].via", "aggregates[0].name", "aggregates[0].using",
                     "aggregates[0].role", "aggregates[0].of", "aggregates[0].by"):
            emitted.add(path)
        checks["crosswalk covers every emitted path"] = (
            {c["spec"].split(" = ")[0] for c in report["crosswalk"]} == emitted)

        # determinism: the same profile serialises identically, key order included
        again, _ = self._compile(profile, binding)
        checks["compilation is byte-stable"] = json.dumps(spec) == json.dumps(again)

        # logical naming is the same document with different column names
        log_spec, _ = self._compile(profile, binding, naming="logical")
        checks["logical naming renames only the columns"] = (
            log_spec["keys"][0]["name"] == "category"
            and log_spec["keys"][1]["name"] == "period"
            and log_spec["aggregates"][0]["name"] == "contract_concentration"
            and log_spec["aggregates"][0]["of"] == "dollars_obligated")

        # ---- a feature + availability + a second (named) target ----
        prof2 = {
            "name": "sales", "version": "1",
            "metrics": [
                {"name": "revenue", "function": "sum", "measure": "amount", "role": "target"},
                {"name": "units", "function": "sum", "measure": "qty", "role": "target"},
                {"name": "promo", "function": "max", "measure": "on_promo",
                 "role": "feature", "availability": "known_ahead"},
            ],
            "dimensions": [{"name": "store"}],
            "time": {"event": "day", "grain": "week"},
            "x-deep": {"series": "store"},
        }
        bind2 = {"source": "pos", "bind": {
            "day": "date", "store": "store_nbr", "amount": "sales_usd",
            "qty": "units_sold", "on_promo": "promo_flag"}}
        spec2, report2 = self._compile(prof2, bind2)
        aggs2 = spec2["aggregates"]
        checks["two targets keep their names"] = (
            aggs2[0]["name"] == "revenue" and aggs2[1]["name"] == "units")
        checks["feature lowered with binding"] = (
            aggs2[2]["name"] == "x_promo" and aggs2[2]["of"] == "promo_flag")
        checks["feature availability copied"] = aggs2[2]["availability"] == "known_ahead"
        checks["roles reach the spec"] = (
            [a["role"] for a in aggs2] == ["target", "target", "feature"])
        checks["list indices densify past the role split"] = len(aggs2) == 3

        # ---- available_from flows through from the binding ----
        bind3 = {**bind2, "available_from": "reported_dt"}
        spec3, _ = self._compile(prof2, bind3)
        checks["available_from becomes validity"] = (
            spec3["validity"] == {"arrival_from": "reported_dt"})

        # ---- filters lower like anything else ----
        prof_f = {**prof2, "filters": [
            {"of": "store", "using": "equals", "value": "12"}]}
        spec_f, _ = self._compile(prof_f, bind2)
        checks["filter lowered through the binding"] = (
            spec_f["filters"] == [{"of": "store_nbr", "using": "equals", "value": "12"}])

        # ---- the pad refuses a conflicting write ----
        pad = Pad()
        pad.write("time_from", "action_dt", "source A")
        pad.write("time_from", "action_dt", "source B")       # same value: a no-op
        conflict = False
        try:
            pad.write("time_from", "reported_dt", "source C")
        except ValueError as e:
            conflict = "source A" in str(e) and "source C" in str(e)
        checks["pad is idempotent on equal writes"] = pad.slots["time_from"][0] == "action_dt"
        checks["pad raises on conflict, naming both"] = conflict
        checks["obligation needs exactly one of logical/literal"] = (
            self._raises(lambda: Obligation("p", logical="a", literal="b"))
            and self._raises(lambda: Obligation("p")))

        # ---- one profile per row of the contract -------------------------
        kinds = {}

        def compiles(kind, prof, bnd):
            try:
                s, _ = self._compile({**prof, "kind": kind}, bnd)
                return s
            except ValueError:
                return None

        cross_bind = {"source": "s", "bind": {
            "hood": "neighbourhood", "price": "sale_price", "beds": "bedrooms",
            "cust": "cust_id", "item": "item_sku", "rating": "stars",
            "spend": "amount", "day": "date"}}

        regress = {"name": "homes", "keys": [{"name": "hood", "from": "hood"}],
                   "metrics": [
                       {"name": "price", "function": "mean", "measure": "price", "role": "target"},
                       {"name": "beds", "function": "mean", "measure": "beds", "role": "feature"}]}
        kinds["regress: no clock at all"] = compiles("regress", regress, cross_bind)

        cluster = {"name": "segments", "keys": [{"name": "cust", "from": "cust"}],
                   "metrics": [
                       {"name": "spend", "function": "sum", "measure": "spend", "role": "feature"},
                       {"name": "orders", "function": "count", "role": "feature"}]}
        kinds["cluster: no target at all"] = compiles("cluster", cluster, cross_bind)

        recommend = {"name": "recs",
                     "keys": [{"name": "cust", "from": "cust"},
                              {"name": "item", "from": "item"}],
                     "metrics": [{"name": "rating", "function": "mean",
                                  "measure": "rating", "role": "target"}]}
        kinds["recommend: two keys"] = compiles("recommend", recommend, cross_bind)

        for label, s in kinds.items():
            checks[label] = s is not None
        if kinds["regress: no clock at all"]:
            checks["regress emits no time_from"] = (
                "time_from" not in kinds["regress: no clock at all"])
        if kinds["cluster: no target at all"]:
            checks["cluster emits only features"] = all(
                a["role"] == "feature"
                for a in kinds["cluster: no target at all"]["aggregates"])
        if kinds["recommend: two keys"]:
            checks["recommend emits two entity keys"] = (
                len(kinds["recommend: two keys"]["keys"]) == 2)

        # ---- validation errors ----
        def fails(prof, bnd, needle, naming="v1"):
            try:
                self._compile(prof, bnd, naming)
                return False
            except ValueError as e:
                return needle in str(e)

        checks["unbound name is rejected"] = fails(
            profile, {"source": "x", "bind": {"action_date": "action_dt"}}, "unbound")
        checks["unknown function is rejected"] = fails(
            {**profile, "metrics": [{"name": "m", "function": "wizardry",
                                     "measure": "obligation", "role": "target"}]},
            binding, "unknown function")
        checks["group fn without 'by' is rejected"] = fails(
            {**profile, "metrics": [{"name": "m", "function": "hhi",
                                     "measure": "obligation", "role": "target"}]},
            binding, "needs a 'by'")
        checks["missing role is rejected"] = fails(
            {**profile, "metrics": [{"name": "m", "function": "sum", "measure": "obligation"}]},
            binding, "needs role")
        checks["bad grain is rejected"] = fails(
            {**profile, "time": {"event": "action_date", "grain": "fortnight"}},
            binding, "grain")
        checks["unknown kind is rejected"] = fails(
            {**profile, "kind": "telepathy"}, binding, "kind")
        checks["grain without event is rejected"] = fails(
            {**profile, "time": {"grain": "quarter"}}, binding, "event is missing")

        # the contract is what catches a question asking for the wrong shape
        checks["forecast without a target is rejected"] = fails(
            {**profile, "metrics": [{"name": "m", "function": "hhi", "measure": "obligation",
                                     "by": "vendor", "role": "feature"}]},
            binding, "requires 'targets'")
        checks["cluster with a target is rejected"] = fails(
            {**profile, "kind": "cluster"}, binding, "forbids 'targets'")
        checks["regress with a clock is rejected"] = fails(
            {**profile, "kind": "regress"}, binding, "forbids 'time_from'")
        # the key-count rule, on a profile that clears the time rules first
        checks["recommend with one key is rejected"] = fails(
            {**regress, "kind": "recommend"}, cross_bind, "entity key")

        # the catalogs must match their embedded fallbacks
        checks["function csv matches embedded fallback"] = (
            function_library.catalog() == function_library.embedded_catalog())
        checks["derive csv matches embedded fallback"] = (
            derive_library.catalog() == derive_library.embedded_catalog())

        passed = sum(1 for v in checks.values() if v)
        return {
            "service": "spec-compiler",
            "passed": passed,
            "total": len(checks),
            "ok": passed == len(checks),
            "checks": checks,
        }

    @staticmethod
    def _raises(fn):
        try:
            fn()
            return False
        except ValueError:
            return True
