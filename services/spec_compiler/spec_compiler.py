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
    from . import derive_library, function_library, task_library
except ImportError:  # pragma: no cover
    import derive_library
    import function_library
    import task_library

GRAINS = derive_library.GRAINS
ROLES = ("target", "feature")            # legacy authoring vocabulary only
AVAIL = task_library.AVAIL
NAMINGS = ("v1", "logical")

# Kinds that were spelled differently before the catalog existed.
KIND_ALIASES = {"anomaly": "detect_anomaly"}

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

# The contract now lives in `task_library` -- one table, shared with the
# profile-generator, so the service that WRITES a task and the service that
# LOWERS it cannot drift apart. It replaced a hand-kept CONTRACT dict here plus
# a separate KNOWN_KINDS/CLOCKED_KINDS pair over there, which is two tables
# describing the same seven questions. Adding an engine is a row.
#
# The mapping layer still never learns what a `kind` is.
KINDS = task_library.kinds()


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


def _clock_of(raw, where):
    """Validate a {event, grain, arrival} clock and return it, or None."""
    if raw in (None, "", {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object with event and grain")
    event = str(raw.get("event") or "").strip()
    grain = str(raw.get("grain") or "").strip().lower()
    arrival = str(raw.get("arrival") or "").strip()
    if grain and grain not in GRAINS:
        raise ValueError(f"{where}.grain must be one of {GRAINS}, got '{raw.get('grain')}'")
    if not event:
        raise ValueError(
            f"{where} is set but {where}.event is missing "
            f"(which logical name carries the timestamp?)")
    out = {"event": event}
    if grain:
        out["grain"] = grain
    if arrival:
        out["arrival"] = arrival
    return out


def lift_task(profile):
    """The profile's task, whether it was authored as one or not.

    `task_field` reads a task-shaped field from either shape. This completes
    that idea: it produces the WHOLE task -- kind, clock, keys, whichever
    pointer names what is predicted, and the covariates -- so everything
    downstream reads one vocabulary and never asks which shape came in.

    The lift is what makes the split adoptable rather than a migration. A flat
    profile carrying `time` and `metrics[].role` is not a second code path; it
    is the same task, spelled the old way, and it compiles to a byte-identical
    SPEC.

    Returns (task, lifted). `lifted` says the flat shape was used, which is
    worth reporting: it is the signal that a producer upstream has not moved.
    """
    native = profile.get("task")
    if native is not None and not isinstance(native, dict):
        raise ValueError("profile.task must be an object when present")
    # Caught here rather than in the catalog, because the catalog only ever
    # sees the lifted task -- a field nothing lifts would vanish silently, and
    # a silently ignored `horizon` on a task is exactly the run-config leak the
    # split exists to prevent.
    for k in (native or {}):
        if k != "kind" and k not in task_library.TASK_FIELDS:
            raise ValueError(
                f"task field '{k}' is not part of the task vocabulary "
                f"({', '.join(task_library.TASK_FIELDS)})")

    raw_kind = str(task_field(profile, "kind") or "forecast").strip().lower()
    kind = KIND_ALIASES.get(raw_kind, raw_kind)
    plan = task_library.plan(kind, "profile.kind")
    task = {"kind": kind}

    # A task authored in full says everything itself; anything it leaves out
    # still falls back to the flat spelling, so the two can be mixed while a
    # producer migrates one field at a time.
    def pick(field):
        return native.get(field) if isinstance(native, dict) and field in native else None

    clock = _clock_of(pick("clock"), "task.clock")
    lifted = clock is None
    if clock is None:
        clock = _clock_of(profile.get("time"), "profile.time")
    if clock:
        task["clock"] = clock

    keys = task_field(profile, "keys")
    names = []
    if isinstance(keys, list) and keys:
        for k in keys:
            if isinstance(k, str) and k.strip():
                names.append(k.strip())            # task form: dimension names
            elif isinstance(k, dict):
                frm = str(k.get("from") or "").strip()
                via = str(k.get("via") or "").strip()
                entry = derive_library.catalog().get(via) if via else None
                ret = entry["returns"] if entry else ""
                if ret == "bucket":
                    continue                       # that is the clock, not a key
                # A key that supplies its own value still identifies a row, it
                # just is not a dimension: `row_number` on data that is already
                # one row per thing has nothing to group by, and dropping it
                # here would tell the catalog the question had no keys at all.
                if frm or via:
                    names.append(frm or via)
    else:
        xdeep = profile.get("x-deep") or profile.get("x_deep") or {}
        if not isinstance(xdeep, dict):
            raise ValueError("profile.x-deep must be an object when present")
        series = str(xdeep.get("series") or "").strip()
        if series:
            names.append(series)
    if names:
        task["keys"] = names

    pointer = task_library.pointer_of(plan) or "target"
    # "Authored" means the task already says what is predicted or carried. A
    # clustering says it by naming covariates and no pointer at all -- that is
    # a complete answer, not a missing one, and reading it as missing sent a
    # perfectly good task down the role-lifting path to be rejected.
    named = next((f for f in task_library.POINTERS if pick(f) is not None), None)
    if named or pick("covariates") is not None:
        if named:
            task[named] = native[named]
        if pick("covariates") is not None:
            task["covariates"] = native["covariates"]
        return task, lifted

    # ---- lift roles off the metrics -----------------------------------
    # A role stamped on a metric grows the metric's enum every time an engine
    # is added -- a recommender wants `signal`, a classifier wants `label`.
    # Inverting it, so the task points AT a metric, freezes the metric schema.
    pointed, covariates = [], []
    for m in (profile.get("metrics") or []):
        if not isinstance(m, dict):
            raise ValueError("each metric must be an object")
        mname = str(m.get("name") or "").strip()
        if not mname:
            raise ValueError("each metric needs a name")
        role = str(m.get("role") or "").strip().lower()
        if role not in ROLES:
            raise ValueError(
                f"metric '{mname}' needs role one of {ROLES}, got '{m.get('role')}' "
                f"(or name it under task.{pointer} / task.covariates)")
        if role == "target":
            pointed.append(mname)
        else:
            covariates.append({
                "metric": mname,
                "availability": str(m.get("availability") or "").strip().lower()
                or "past_only"})
    if pointed:
        task[pointer] = pointed
    if covariates:
        task["covariates"] = covariates
    return task, True


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

        `profile.keys` is the general form: objects that name their own output
        column. `task.keys` is the short form, bare dimension names, and the
        named blocks are sugar over that -- an x-deep series becomes one key,
        and a time grain becomes a bin: key. Nothing downstream can tell which
        form was authored.
        """
        # A top-level keys[] outranks task.keys when both are present: the
        # author who spelled out the output columns has said something the
        # short form cannot say, and a naming rule must not overwrite it.
        explicit = profile.get("keys")
        raw = (explicit if isinstance(explicit, list) and explicit
               else task_field(profile, "keys"))

        # -- short form: bare dimension names, so the naming rule applies ----
        # One entity key keeps the shape a single-series forecast has always
        # emitted; two or more had no legacy spelling to preserve, so they are
        # prefixed rather than numbered, which reads in a frame header.
        if isinstance(raw, list) and raw and all(isinstance(k, str) for k in raw):
            names = [k.strip() for k in raw if k.strip()]
            multi = len(names) > 1
            out = []
            for n in names:
                out.append({
                    "name": (("k_" + n) if multi else "series_id") if naming == "v1" else n,
                    "from": n, "via": "", "origin": f"task.keys = {n}"})
            grain = str((task_field(profile, "clock") or {}).get("grain")
                        if isinstance(task_field(profile, "clock"), dict)
                        else (profile.get("time") or {}).get("grain") or "").strip().lower()
            if grain:
                out.append({"name": "t" if naming == "v1" else "period",
                            "from": "", "via": "bin:" + grain,
                            "origin": f"task.clock.grain = {grain}"})
            return out

        if isinstance(raw, list) and raw:
            out = []
            for i, k in enumerate(raw):
                if not isinstance(k, dict):
                    raise ValueError(
                        f"profile.keys[{i}] must be an object, or the whole list "
                        f"must be bare dimension names (task.keys form)")
                name = str(k.get("name") or "").strip()
                frm = str(k.get("from") or "").strip()
                via = str(k.get("via") or "").strip()
                if not name:
                    raise ValueError(f"profile.keys[{i}].name is required")
                entry = derive_library.resolve(via, f"profile.keys[{i}].via") \
                    if via else None
                # Which derives may omit `from` is a property of the derive, not
                # a prefix on its name: a bin: spine reads the source's own event
                # time, and a position key reads the row's index. Matching on the
                # string "bin:" made that a naming convention rather than a fact
                # the library states about itself.
                self_supplying = bool(entry) and entry["returns"] in ("bucket",
                                                                      "position")
                if not frm and not self_supplying:
                    raise ValueError(
                        f"profile.keys[{i}] needs a 'from' (only a key whose "
                        f"derive supplies its own value may omit it)")
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

    def _walk(self, profile, task, binding, lib, naming):
        """Yield every obligation the profile raises, in a fixed order.

        Lists in list order, catalog `needs` in catalog order. Nothing here
        iterates a set: two runs of the same profile must produce byte-identical
        JSON, not merely equal dicts.

        Reads the lifted `task`, never the flat blocks: by this point there is
        one vocabulary, and which shape was authored is already forgotten.
        """
        # -- P3 the clock column (the keys carry the grain) ------------------
        clock = task.get("clock") or {}
        event = str(clock.get("event") or "").strip()
        if event:
            yield Obligation("time_from", logical=event, origin="task.clock.event")

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
        # Which metric is predicted comes from the task's pointer, not from a
        # role on the metric. It decides the v1 column name and nothing else.
        pointer = task_library.pointer_of(task_library.plan(task["kind"]))
        pointed = task_library.as_list(task.get(pointer)) if pointer else []
        cov = {}
        for c in (task.get("covariates") or []):
            if isinstance(c, dict):
                cov[str(c.get("metric") or "").strip()] = \
                    str(c.get("availability") or "").strip().lower() or "past_only"
        single_target = len(pointed) == 1
        for j, m in enumerate(metrics):
            if not isinstance(m, dict):
                raise ValueError("each metric must be an object")
            mname = str(m.get("name") or "").strip()
            if not mname:
                raise ValueError("each metric needs a name")
            if mname not in pointed and mname not in cov:
                raise ValueError(
                    f"metric '{mname}' is declared but the task never names it "
                    f"(put it under task.{pointer or 'covariates'} or task.covariates)")
            fn = str(m.get("function") or "").strip()
            if not fn:
                raise ValueError(f"metric '{mname}' has no function")
            entry = lib.get(fn)
            if entry is None:
                raise ValueError(
                    f"metric '{mname}' uses unknown function '{fn}' "
                    f"(not in the function library: {sorted(lib)})")

            if naming == "v1":
                out = ("y" if (mname in pointed and single_target)
                       else mname if mname in pointed else "x_" + mname)
            else:
                out = mname
            base = f"aggregates[{j}]"
            yield Obligation(f"{base}.name", literal=out, origin=f"metric {mname}")
            yield Obligation(f"{base}.using", literal=fn,
                             origin=f"metric {mname}.function")
            # `role` is deliberately NOT emitted. Which column you predict is a
            # modelling decision that changes between runs of the same frame, so
            # it belongs to the run config. It is still read from the profile
            # here, to pick the v1 column names and to scaffold that run config.

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

            # `availability` DOES reach the SPEC, because it is a fact about the
            # data rather than about the modelling task: whether a value is
            # knowable at prediction time is what decides which bucket a late
            # one lands in. A predicted column is a label and always describes
            # its own period, so it declares nothing.
            if mname in cov:
                avail = cov[mname]
                if avail not in AVAIL:
                    raise ValueError(
                        f"covariate '{mname}' availability must be one of {AVAIL}, "
                        f"got '{avail}'")
                yield Obligation(f"{base}.availability", literal=avail,
                                 origin=f"task.covariates '{mname}'.availability")

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
        # Only meaningful when there are periods to be re-filed between. A
        # clustering has no clock, so "when did this become knowable" has no
        # bucket to answer with, and emitting a guard that can never fire is
        # noise in the SPEC. The old contract said this per KIND, which forbade
        # it to a classification even when that classification had a clock.
        arrival_logical = str(clock.get("arrival") or "").strip()
        arrival_physical = (str(binding.get("available_from") or "").strip()
                            if clock else "")
        if arrival_logical:
            yield Obligation("validity.arrival_from", logical=arrival_logical,
                             origin="task.clock.arrival")
        elif arrival_physical:
            yield Obligation("validity.arrival_from", literal=arrival_physical,
                             origin="binding.available_from")

    # ----------------------------------------------------------- run config

    @staticmethod
    def _run_config(profile, task, spec, kind, naming):
        """The modelling half: which emitted columns are the target(s).

        Column names here are the SPEC's output names, so this is already in
        the frame's vocabulary — the engine adapter never has to look back at
        the profile.

        It speaks the run config's vocabulary rather than the kind's: an engine
        reads `target` whether the task called the thing a signal or a label.
        Which of those three the question used is still reported, because it is
        the difference between scoring an interaction and predicting a class.
        """
        pointer = task_library.pointer_of(task_library.plan(kind))
        pointed = set(task_library.as_list(task.get(pointer))) if pointer else set()
        known_ahead_of = {
            str(c.get("metric") or "").strip():
                str(c.get("availability") or "").strip().lower() == "known_ahead"
            for c in (task.get("covariates") or []) if isinstance(c, dict)}

        targets, features, known_ahead = [], [], []
        for m, agg in zip([m for m in (profile.get("metrics") or [])
                           if isinstance(m, dict)], spec.get("aggregates") or []):
            mname = str(m.get("name") or "").strip()
            col = agg.get("name")
            if not col:
                continue
            if mname in pointed:
                targets.append(col)
            else:
                features.append(col)
                if known_ahead_of.get(mname):
                    known_ahead.append(col)

        rc = {"kind": kind, "pointer": pointer, "targets": targets,
              "features": features, "known_ahead": known_ahead}
        if len(targets) == 1:
            rc["target"] = targets[0]
        return rc

    # ------------------------------------------------------------ contract

    @staticmethod
    def _check(spec, task, plan, pad):
        """Hold the emitted SPEC against the task it came from.

        `task_library.check` already held the PLAN against its kind's row --
        that a forecast names a target, that a recommender takes exactly two
        keys, that a clustering was not handed something to predict. This is
        the other half: a template fails loudly, because a slot you forgot to
        fill is a visible empty box, but a walk fails by OMISSION, which is
        quiet. This is what closes that gap, and it is the reason the walk is
        allowed to be permissive.

        Which keys are the clock comes from what the derive returns, not from
        the string 'bin:' -- the same fact the library states about itself that
        `_keys_of` reads.
        """
        dlib = derive_library.catalog()

        def returns(k):
            entry = dlib.get(str(k.get("via") or "").strip())
            return entry["returns"] if entry else ""

        keys = spec.get("keys") or []
        binned = [k for k in keys if returns(k) == "bucket"]
        entity = [k for k in keys if returns(k) != "bucket"]
        clock = task.get("clock") or {}

        if clock:
            if not spec.get("time_from"):
                raise ValueError(
                    "the task declares a clock, but nothing emitted 'time_from'")
            if clock.get("grain") and not binned:
                raise ValueError(
                    f"the task's clock is grained '{clock['grain']}', but no key bins it")
        else:
            if spec.get("time_from"):
                raise ValueError(
                    f"the task declares no clock, but {pad.origin_of('time_from')} "
                    f"emitted 'time_from'")
            if binned:
                raise ValueError(
                    "the task declares no clock, but a key bins one")

        lo, hi = plan["keys_min"], plan["keys_max"]
        if not lo <= len(entity) <= hi:
            raise ValueError(
                f"kind '{task['kind']}' needs {lo}..{hi} entity key(s), "
                f"got {len(entity)} ({[k.get('name') for k in entity]})")

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

        # The task is lifted first, so everything after it reads one vocabulary
        # whether the profile was authored flat or split.
        task, lifted = lift_task(profile)
        kind = task["kind"]
        report["kind"] = kind
        report["task"] = task
        report["lifted_from_flat"] = lifted

        metrics = profile.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise ValueError("profile.metrics must be a non-empty list")
        metric_names = [str(m.get("name") or "").strip()
                        for m in metrics if isinstance(m, dict)]
        dim_names = [str(d.get("name") or "").strip()
                     for d in (profile.get("dimensions") or []) if isinstance(d, dict)]
        # The catalog's dimension check is deliberately NOT run here. The
        # profile-generator runs it at finalize, where a human is about to sign
        # and can fix the profile. By this point the binding is in hand, so a
        # key naming an undeclared dimension still resolves to a real column and
        # produces a correct frame -- refusing it would reject a profile that
        # works, and every other catalog rule still applies.
        plan, kind, pointer, pointed = task_library.check(task, metric_names)

        for k in self._keys_of(profile, naming):
            if k["from"] and dim_names and k["from"] not in dim_names:
                report["warnings"].append(
                    f"key source '{k['from']}' is not listed under dimensions")
        if binding.get("available_from") and not task.get("clock"):
            report["warnings"].append(
                f"binding.available_from is set, but a '{kind}' has no clock; "
                f"there are no periods to re-file between, so no validity "
                f"block was emitted")
        report["stages"].append(
            f"lift+check: '{kind}' task, {len(pointed)} predicted, "
            f"{len(task.get('covariates') or [])} covariate(s)"
            + (" (lifted from a flat profile)" if lifted else ""))

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
        for ob in self._walk(profile, task, binding, lib, naming):
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

        # -- stage 5: check the emitted document -------------------------
        self._check(spec, task, plan, pad)

        # -- stage 6: scaffold the run config ----------------------------
        # The SPEC says how the frame is built; this says what to do with it.
        # Splitting them is what lets the same frame answer a different
        # question — swap the target here and no cell of the frame changes.
        report["run_config"] = self._run_config(profile, task, spec, kind, naming)
        report["stages"].append(f"check: SPEC satisfies the '{kind}' task")
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
            "aggregates": [{"name": "y", "using": "hhi",
                            "of": "dollars_obligated", "by": "recipient_parent"}],
        }
        checks["compiled SPEC matches the design doc"] = spec == expected
        checks["the SPEC says how, the run config says what"] = (
            "role" not in spec["aggregates"][0]
            and report["run_config"]["target"] == "y")

        # the crosswalk is generated, so it covers the document by construction
        emitted = set()
        for path in ("time_from", "keys[0].name", "keys[0].from", "keys[1].name",
                     "keys[1].via", "aggregates[0].name", "aggregates[0].using",
                     "aggregates[0].of", "aggregates[0].by"):
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
        checks["role does NOT reach the spec"] = (
            not any("role" in a for a in aggs2))
        checks["roles reach the run config instead"] = (
            report2["run_config"]["targets"] == ["revenue", "units"]
            and report2["run_config"]["features"] == ["x_promo"])
        checks["single target also exposed as run_config.target"] = (
            report["run_config"].get("target") == "y")
        checks["run config carries the kind"] = report2["run_config"]["kind"] == "forecast"
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
            # The SPEC cannot say "these are features" any more, and does not
            # need to: a clustering frame is just columns. What must still hold
            # is that nothing was nominated as a target.
            checks["cluster emits aggregates with no role"] = all(
                "role" not in a
                for a in kinds["cluster: no target at all"]["aggregates"])
        if kinds["recommend: two keys"]:
            checks["recommend emits two entity keys"] = (
                len(kinds["recommend: two keys"]["keys"]) == 2)

        # ---- the same question, authored as a core and a task -------------
        # The migration rests on this: no producer has to change on the day the
        # compiler does, and one that has changed gets the same frame.
        native = {
            "name": "supplier_concentration", "version": "3",
            "datasets": [{"name": "awards"}],
            "metrics": [{"name": "contract_concentration", "function": "hhi",
                         "measure": "obligation", "by": "vendor"}],
            "dimensions": [{"name": "category"}, {"name": "vendor"}],
            "task": {"kind": "forecast",
                     "clock": {"event": "action_date", "grain": "quarter"},
                     "keys": ["category"],
                     "target": "contract_concentration"},
        }
        nspec, nrep = self._compile(native, binding)
        checks["a task-shaped profile compiles byte-identically"] = (
            json.dumps(nspec) == json.dumps(spec))
        checks["no metric in a task-shaped profile carries a role"] = not any(
            "role" in m or "availability" in m for m in native["metrics"])
        checks["the run config is the same either way"] = (
            nrep["run_config"] == report["run_config"])
        checks["a flat profile reports that it was lifted"] = (
            report["lifted_from_flat"] is True
            and nrep["lifted_from_flat"] is False)
        checks["the report echoes the lifted task"] = (
            report["task"] == {"kind": "forecast",
                               "clock": {"event": "action_date", "grain": "quarter"},
                               "keys": ["category"],
                               "target": ["contract_concentration"]})

        # a covariate carries availability in the task, not on the metric
        native_cov = {
            "name": "sales", "version": "1",
            "metrics": [{"name": "revenue", "function": "sum", "measure": "amount"},
                        {"name": "promo", "function": "max", "measure": "on_promo"}],
            "dimensions": [{"name": "store"}],
            "task": {"kind": "forecast", "clock": {"event": "day", "grain": "week"},
                     "keys": ["store"], "target": "revenue",
                     "covariates": [{"metric": "promo", "availability": "past_only"}]},
        }
        ncov, _ = self._compile(native_cov, bind2)
        checks["task covariates lower to availability"] = (
            ncov["aggregates"][1]["availability"] == "past_only"
            and "availability" not in ncov["aggregates"][0])

        # the pointer a kind uses is the kind's business, not the author's
        native_sig = {
            "name": "recs",
            "metrics": [{"name": "rating", "function": "mean", "measure": "rating"}],
            "dimensions": [{"name": "cust"}, {"name": "item"}],
            "task": {"kind": "recommend", "keys": ["cust", "item"], "signal": "rating"}}
        sig_spec, sig_rep = self._compile(native_sig, cross_bind)
        checks["a signal is a pointer like any other"] = (
            sig_spec["aggregates"][0]["name"] == "y"
            and sig_rep["run_config"]["target"] == "y")
        checks["the run config reports which pointer was used"] = (
            sig_rep["run_config"]["pointer"] == "signal"
            and report["run_config"]["pointer"] == "target")

        # A leakage guard needs a clock to be a guard against anything. The old
        # contract said this per KIND, which forbade it to a classification even
        # when that classification had a clock.
        clockless_av, cl_rep = self._compile(
            native_sig, {**cross_bind, "available_from": "reported_dt"})
        checks["a clockless question emits no validity"] = (
            "validity" not in clockless_av)
        checks["and says so rather than dropping it silently"] = any(
            "no periods to re-file between" in w for w in cl_rep["warnings"])
        clocked_cl, _ = self._compile(
            {"name": "risk",
             "metrics": [{"name": "risk", "function": "label", "measure": "rating"}],
             "dimensions": [{"name": "cust"}],
             "task": {"kind": "classify", "keys": ["cust"], "label": "risk",
                      "clock": {"event": "day", "grain": "week"}}},
            {**cross_bind, "available_from": "reported_dt"})
        checks["a clocked classification may still have one"] = (
            clocked_cl["validity"] == {"arrival_from": "reported_dt"})

        # `anomaly` was the name before the catalog had a row for it
        checks["the old kind name still resolves"] = (
            self._compile({**prof2, "kind": "anomaly"}, bind2)[1]["kind"]
            == "detect_anomaly")
        checks["every catalogued kind is reachable"] = (
            set(task_library.kinds()) == {"forecast", "detect_anomaly", "classify",
                                          "regress", "cluster", "recommend", "rank"})

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

        # the catalog is what catches a question asking for the wrong shape.
        # It says the same three things the CONTRACT table used to, in the
        # task's vocabulary rather than the SPEC's: a required field is named
        # `target`, not `targets`, and a field a kind cannot use is not
        # "forbidden" but meaningless -- it HAS NO CONCEPT of it.
        checks["forecast without a target is rejected"] = fails(
            {**profile, "metrics": [{"name": "m", "function": "hhi", "measure": "obligation",
                                     "by": "vendor", "role": "feature"}]},
            binding, "requires 'target'")
        checks["cluster with a target is rejected"] = fails(
            {**profile, "kind": "cluster"}, binding, "has no concept of 'target'")
        # A kind that merely PERMITS a clock now keeps it -- regress over time
        # is a real question, and the old table forbade it by hand. What is
        # still rejected is a kind with no concept of one at all.
        checks["regress with a clock is allowed"] = (
            compiles("regress", profile, binding) is not None)
        checks["recommend with a clock is rejected"] = fails(
            {**recommend, "kind": "recommend",
             "time": {"event": "day", "grain": "week"}},
            {**cross_bind, "bind": {**cross_bind["bind"], "day": "date"}},
            "has no concept of 'clock'")
        # the key-count rule, on a profile that clears the field rules first
        checks["recommend with one key is rejected"] = fails(
            {**regress, "kind": "recommend"}, cross_bind, "needs 2..2 key(s)")

        # A field the kind has no concept of is an error, not an ignored hint:
        # an optional field would mean every engine had to know to skip it, and
        # nothing would catch a recommender that set a grain by mistake.
        checks["a task field outside the vocabulary is rejected"] = fails(
            {**native_sig, "task": {**native_sig["task"], "horizon": 8}},
            cross_bind, "not part of the task vocabulary")
        checks["a pointer naming an undeclared metric is rejected"] = fails(
            {**native_sig, "task": {**native_sig["task"], "signal": "nonesuch"}},
            cross_bind, "not declared")
        # A key naming an undeclared dimension is the profile-generator's to
        # reject at finalize, while a human can still fix it. Here the binding
        # is in hand, so it resolves to a real column: warn, do not refuse.
        _, warn_rep = self._compile(
            {**native_sig, "dimensions": [{"name": "cust"}],
             "task": {**native_sig["task"], "keys": ["cust", "item"]}}, cross_bind)
        checks["an undeclared dimension warns rather than refuses"] = any(
            "not listed under dimensions" in w for w in warn_rep["warnings"])
        checks["a metric cannot be both predicted and carried"] = fails(
            {**native_sig, "task": {**native_sig["task"],
                                    "covariates": [{"metric": "rating"}]}},
            cross_bind, "both predicted and a covariate")
        checks["a metric the task never names is rejected"] = fails(
            {**native_sig, "metrics": native_sig["metrics"] + [
                {"name": "spare", "function": "count"}]},
            cross_bind, "never names it")
        checks["a bad covariate availability is rejected"] = fails(
            {**native_cov, "task": {**native_cov["task"],
                                    "covariates": [{"metric": "promo",
                                                    "availability": "someday"}]}},
            bind2, "availability must be one of")

        # the catalogs must match their embedded fallbacks
        checks["task csv matches embedded fallback"] = (
            task_library.catalog() == task_library.embedded_catalog())
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
