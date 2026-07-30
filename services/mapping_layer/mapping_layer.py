"""mapping-layer: the pipeline MAPPING LAYER as an SPL internal service.

Runs an approved SPEC against raw tabular rows and emits the canonical frame
the decision engine eats:

    filter -> join sources (on time) -> resolve series -> bucket time -> compute y1..yn

Input   request.data        raw rows (list of dicts); with multiple sources each
                            row carries a discriminator column (source_from)
        request.parameters  the approved SPEC, flattened (see schema())
Output  RunResponse.data    frame rows: {series_id?, t, <target cols>, x_<feature>...}
        RunResponse.metadata row counts, drop reasons, join gaps per target,
                            each feature's availability flag, and (when a source
                            declares available_from) a leakage guard that re-files
                            late-arriving past_only covariates forward

Successor of frame-executor 1.0.0 (v1 flat single-target specs still run
unchanged). Domain-blind by design: it only knows the column names handed to it
in parameters, never what they mean. Stdlib only - no package_dependencies.
"""

import asyncio
from datetime import date, datetime

from spl.core.base_service.base_service_class import BaseService
from spl.core.service_types import ParameterEnum

try:  # sibling modules: package-relative when deployed, flat when run locally
    from . import derive_library, function_library
except ImportError:  # pragma: no cover
    import derive_library
    import function_library

GRAINS = derive_library.GRAINS
DERIVES = tuple(n for n, e in derive_library.embedded_catalog().items()
                if e["returns"] == "scalar")
AVAIL = ("known_ahead", "past_only")
ROLES = ("target", "feature")
FILTER_OPS = ("equals", "notEquals", "greaterThan", "lessThan", "contains", "isEmpty", "isNotEmpty")
DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%Y%m%d")

# series bucket key for rows from sources that don't carry the series column;
# their values are broadcast onto every concrete series at the same t
_BCAST = "\x00broadcast"


class MappingLayerService(BaseService):

    # ------------------------------------------------------------------ run

    async def _run(self, request: BaseService.RunRequest) -> BaseService.RunResponse:
        rows = request.data or []
        try:
            spec = self._resolve_spec(request.parameters or {})
        except ValueError as e:
            self.logger.error(f"mapping-layer: invalid spec: {e}")
            return BaseService.RunResponse(data=[{"error": f"invalid spec: {e}"}])

        frame, meta = await asyncio.to_thread(self._execute, rows, spec)

        dropped = meta["rows_dropped"]
        if dropped["bad_time"] or dropped["bad_measure"] or dropped["unknown_source"]:
            self.logger.warning(
                f"mapping-layer: dropped {dropped['bad_time']} row(s) with an unparseable "
                f"time, {dropped['bad_measure']} non-numeric measure value(s), "
                f"{dropped['unknown_source']} row(s) with an undeclared source id"
            )
        self.logger.info(
            f"mapping-layer: {meta['rows_in']} raw row(s) -> {meta['frame_rows']} frame "
            f"row(s), {len(spec['targets'])} target(s), grain '{spec['time_grain']}'"
        )
        return BaseService.RunResponse(data=frame, metadata=meta)

    # ------------------------------------------------------------- the spec

    @staticmethod
    def _param_root(parameters):
        inner = parameters.get("serviceInstructions")
        return inner if isinstance(inner, dict) else parameters

    @staticmethod
    def _to_v2(p):
        """Normalise a legacy flat SPEC into the positioned v2 vocabulary.

        v2 declares every frame column by name. v1 hard-coded them
        (series_id / t / y / x_<name>), so translating rather than branching
        keeps a single execution path and makes v1 exactly a dialect of v2 --
        the legacy names are simply what this translation writes into `out`.
        """
        if p.get("keys") is not None or p.get("aggregates") is not None:
            return p                                        # already v2

        v2 = {k: v for k, v in p.items()
              if k in ("sources", "source_from", "time_from", "available_from", "validity")}

        keys = []
        series_from = str(p.get("series_from") or "").strip()
        if series_from:
            keys.append({"name": "series_id", "from": series_from,
                         "prefix": str(p.get("series_prefix") or ""),
                         "aliases": p.get("series_aliases") or []})
        grain = str(p.get("time_grain") or "week").strip().lower()
        if grain not in GRAINS:
            raise ValueError(f"time_grain must be one of {GRAINS}, got '{grain}'")
        keys.append({"name": "t", "via": "bin:" + grain})
        v2["keys"] = keys

        raw_targets = p.get("targets") or []
        if not raw_targets:
            # v1 frame-executor compatibility: flat single-target spec
            raw_targets = [{"name": "y",
                            "fn": str(p.get("target_metric") or "sum"),
                            "measure": str(p.get("target_measure") or "")}]
        single = len(raw_targets) == 1
        aggregates, derives = [], []
        for i, tg in enumerate(raw_targets):
            if not isinstance(tg, dict):
                raise ValueError(f"targets[{i}] must be an object")
            name = str(tg.get("name") or "").strip() or ("y" if single else "")
            aggregates.append({
                "name": name, "out": name, "role": "target", "ord": i,
                "using": tg.get("fn"), "_alias": tg.get("metric"), "_default": "sum",
                "of": tg.get("measure"), "by": tg.get("entity"), "source": tg.get("source"),
            })
        for i, f in enumerate(p.get("features") or []):
            if not isinstance(f, dict):
                raise ValueError(f"features[{i}] must be an object")
            name = str(f.get("name") or "").strip()
            if not name:
                raise ValueError(f"features[{i}].name is required")
            derive = str(f.get("derive") or "").strip().lower()
            if derive:
                if str(f.get("measure") or "").strip():
                    raise ValueError(f"feature '{name}': set 'measure' OR 'derive', not both")
                derives.append({"name": name, "out": "x_" + name, "scope": "key",
                                "from": "t", "via": derive, "ord": i,
                                "availability": f.get("availability")})
                continue
            aggregates.append({
                "name": name, "out": "x_" + name, "role": "feature", "ord": i,
                "using": f.get("fn"), "_alias": f.get("metric"), "_default": "max",
                "of": f.get("measure"), "by": f.get("entity"), "source": f.get("source"),
                "availability": f.get("availability"),
            })
        v2["aggregates"] = aggregates
        v2["derives"] = derives

        filters = []
        for i, f in enumerate(p.get("filters") or []):
            if not isinstance(f, dict):
                raise ValueError(f"filters[{i}] must be an object")
            col = f.get("of") if str(f.get("of") or "").strip() else f.get("column")
            if not str(col or "").strip():
                raise ValueError(f"filters[{i}] needs a 'column'")
            filters.append({"of": col,
                            "using": f.get("using") or f.get("operator") or "equals",
                            "value": f.get("value"),
                            "applied_upstream": f.get("applied_upstream")})
        v2["filters"] = filters
        return v2

    def _resolve_spec(self, parameters):
        p = self._to_v2(self._param_root(parameters))

        raw_keys = p.get("keys")
        if not isinstance(raw_keys, list) or not raw_keys:
            raise ValueError("keys is required: at least one column identifies a frame row")

        # Only a key that bins the sources' own event time needs a time column.
        # A cross-sectional frame legitimately has no clock at all.
        needs_time = any(
            isinstance(k, dict) and str(k.get("via") or "").startswith("bin:")
            and not str(k.get("from") or "").strip()
            for k in raw_keys)

        # -- P1 sources ----------------------------------------------------
        top_time = str(p.get("time_from") or "").strip()
        validity = p.get("validity") if isinstance(p.get("validity"), dict) else {}
        top_avail = (str(p.get("available_from") or "").strip()
                     or str(validity.get("arrival_from") or "").strip())
        sources = []
        for i, s in enumerate(p.get("sources") or []):
            if not isinstance(s, dict):
                raise ValueError(f"sources[{i}] must be an object")
            sid = str(s.get("id") or "").strip()
            if not sid:
                raise ValueError(f"sources[{i}].id is required")
            tf = str(s.get("time_from") or "").strip() or top_time
            if needs_time and not tf:
                raise ValueError(f"source '{sid}' needs a time_from (its own or the top-level one)")
            af = str(s.get("available_from") or "").strip() or top_avail
            sources.append({"id": sid, "time_from": tf, "available_from": af})
        if not sources:
            if needs_time and not top_time:
                raise ValueError("time_from is required (which column carries the timestamp)")
            sources = [{"id": "", "time_from": top_time, "available_from": top_avail}]
        ids = [s["id"] for s in sources]
        id_set = set(ids)
        if len(id_set) != len(ids):
            raise ValueError("source ids must be unique")
        multi = len(sources) > 1

        source_from = str(p.get("source_from") or "").strip()
        if multi and not source_from:
            raise ValueError(
                "source_from is required with multiple sources "
                "(the column tagging each row with its source id)")

        def split(colname, what):
            """'power.value' -> ('power', 'value'); unqualified allowed only single-source."""
            col = str(colname or "").strip()
            if "." in col:
                pre, rest = col.split(".", 1)
                if pre in id_set and rest:
                    return pre, rest
            if multi:
                raise ValueError(
                    f"{what}: with multiple sources, qualify columns as "
                    f"<source>.<column> (got '{col}')")
            return sources[0]["id"], col

        # -- P5 keys (with P3 derives) -------------------------------------
        # What identifies one frame row. Order is both the composite-key order
        # and the sort order. A `bin:` key with no `from` is the shared time
        # spine: it reads whichever column each source declared as its clock.
        keys, key_names, spine = [], set(), []
        for i, k in enumerate(raw_keys):
            if not isinstance(k, dict):
                raise ValueError(f"keys[{i}] must be an object")
            name = str(k.get("name") or "").strip()
            if not name:
                raise ValueError(f"keys[{i}].name is required")
            if name in key_names:
                raise ValueError(f"duplicate key name '{name}'")
            key_names.add(name)
            via = str(k.get("via") or "").strip()
            raw_from = str(k.get("from") or "").strip()
            vkind = "raw"
            if via:
                entry = derive_library.resolve(via, f"keys[{i}].via")
                if entry["scope"] != "key":
                    raise ValueError(
                        f"keys[{i}].via '{via}' has {entry['scope']} scope; "
                        f"a key needs a key-scope derive")
                vkind = entry["returns"]
            is_spine = vkind == "bucket" and not raw_from
            if not is_spine and not raw_from:
                raise ValueError(
                    f"keys[{i}].from is required (only a bin: key may omit it, "
                    f"to bin each source's own event time)")
            ksrc, kcol = (None, "") if is_spine else split(raw_from, f"keys[{i}].from")
            keys.append({
                "name": name, "source": ksrc, "column": kcol, "via": via,
                "vkind": vkind, "spine": is_spine,
                "prefix": str(k.get("prefix") or ""),
                "aliases": {str(a.get("value")): str(a.get("replace_with"))
                            for a in (k.get("aliases") or [])
                            if isinstance(a, dict) and a.get("value") is not None
                            and a.get("replace_with") is not None},
            })
            if is_spine:
                spine.append(i)

        time_grain = keys[spine[0]]["via"][4:] if spine else ""
        t_key = keys[spine[0]]["name"] if spine else None

        # -- the function library -----------------------------------------
        lib = function_library.catalog()

        def resolve_fn(raw_fn, raw_metric, default, what):
            """Resolve a target/feature to a catalogued, implemented function.

            `fn` is the field going forward; `metric` is the v1 alias. Names not
            in the library, or catalogued-but-not-built, are rejected with a
            clear message (the flowchart's 'the library flags it so we build it').
            """
            picked = raw_fn if (raw_fn is not None and str(raw_fn).strip()) else raw_metric
            if (picked is None or not str(picked).strip()) and not default:
                raise ValueError(
                    f"{what}: needs a 'using' function "
                    f"(known: {', '.join(sorted(lib))})")
            name = str(picked or default).strip().lower()
            meta = lib.get(name)
            if meta is None:
                raise ValueError(
                    f"{what}: '{name}' is not in the function library "
                    f"(known: {', '.join(sorted(lib))}). Add it to "
                    f"{function_library.CATALOG_FILENAME} and implement it in function_library.py")
            if not function_library.is_implemented(name):
                raise ValueError(
                    f"{what}: '{name}' is catalogued on the '{meta['shelf']}' shelf but not "
                    f"yet implemented — build it in function_library.py before using it")
            return name, meta

        def qualify(raw_col, explicit, what):
            """Resolve a column to (source, column), honoring an explicit source."""
            raw_col = str(raw_col or "").strip()
            if explicit:
                col = raw_col
                if "." in raw_col:
                    pre, rest = raw_col.split(".", 1)
                    if pre in id_set:
                        if pre != explicit:
                            raise ValueError(
                                f"{what}: qualified to '{pre}' but source says '{explicit}'")
                        col = rest
                return explicit, col
            return split(raw_col, what)

        # -- P6 aggregates --------------------------------------------------
        # One entry per output column that is not a key. `role` is a field, not
        # a section, which is what lets a frame have several targets, or none
        # at all (clustering) without the executor knowing what a kind is.
        targets, features, seen = [], [], set(key_names)
        for i, a in enumerate(p.get("aggregates") or []):
            if not isinstance(a, dict):
                raise ValueError(f"aggregates[{i}] must be an object")
            name = str(a.get("name") or "").strip()
            if not name:
                raise ValueError(f"aggregates[{i}].name is required")
            role = str(a.get("role") or "target").strip().lower()
            if role not in ROLES:
                raise ValueError(f"aggregate '{name}': role must be one of {ROLES}, got '{role}'")
            out = str(a.get("out") or "").strip() or name
            if out in key_names:
                raise ValueError(f"aggregate column '{out}' collides with a key column")
            if out in seen:
                raise ValueError(f"duplicate output column '{out}'")
            seen.add(out)
            fn_name, fmeta = resolve_fn(a.get("using"), a.get("_alias"),
                                        a.get("_default"), f"{role} '{name}'")
            needs = fmeta["needs"]
            explicit = str(a.get("source") or "").strip()
            if explicit and explicit not in id_set:
                raise ValueError(f"{role} '{name}': unknown source '{explicit}'")
            measure = str(a.get("of") or "").strip()
            if "measure" in needs:
                if not measure:
                    raise ValueError(f"{role} '{name}': function '{fn_name}' needs a measure")
                src, col = qualify(measure, explicit, f"{role} '{name}' measure")
            elif measure:
                src, col = qualify(measure, explicit, f"{role} '{name}' measure")
            elif explicit:
                src, col = explicit, ""
            elif not multi:
                src, col = sources[0]["id"], ""
            else:
                raise ValueError(
                    f"{role} '{name}': function '{fn_name}' with multiple sources needs a 'source'")
            entity_col = ""
            if "entity" in needs:
                ent_raw = str(a.get("by") or a.get("entity") or "").strip()
                if not ent_raw:
                    raise ValueError(f"{role} '{name}': function '{fn_name}' needs an 'entity' column")
                _, entity_col = qualify(ent_raw, src, f"{role} '{name}' entity")
            item = {"name": name, "out": out, "fn": fn_name, "kind": fmeta["kind"],
                    "shelf": fmeta["shelf"], "source": src, "column": col,
                    "entity_col": entity_col, "ord": int(a.get("ord") or i)}
            if role == "target":
                targets.append(item)
            else:
                availability = str(a.get("availability") or "past_only").strip().lower()
                if availability not in AVAIL:
                    raise ValueError(
                        f"feature '{name}': availability must be one of {AVAIL}, got '{availability}'")
                item.update({"derive": "", "derive_key": None, "availability": availability})
                features.append(item)

        # -- P3 key-scope derives -------------------------------------------
        # Calendar projections of a binned key. They read the bucket's sort
        # value, so they are always defined and never count toward whether a
        # row has any measured content.
        for i, d in enumerate(p.get("derives") or []):
            if not isinstance(d, dict):
                raise ValueError(f"derives[{i}] must be an object")
            name = str(d.get("name") or "").strip()
            if not name:
                raise ValueError(f"derives[{i}].name is required")
            out = str(d.get("out") or "").strip() or name
            if out in key_names:
                raise ValueError(f"derive column '{out}' collides with a key column")
            if out in seen:
                raise ValueError(f"duplicate output column '{out}'")
            seen.add(out)
            via = str(d.get("via") or "").strip()
            entry = derive_library.resolve(via, f"derives[{i}].via")
            scope = str(d.get("scope") or entry["scope"]).strip().lower()
            if scope != entry["scope"]:
                raise ValueError(
                    f"derives[{i}] declares {scope} scope but '{via}' is "
                    f"{entry['scope']} scope in the derive library")
            if scope != "key":
                raise ValueError(
                    f"derives[{i}]: {scope}-scope derives are catalogued but not "
                    f"executable yet - build one in derive_library.py first")
            src_key = str(d.get("from") or "").strip() or (t_key or "")
            idx = next((j for j, k in enumerate(keys) if k["name"] == src_key), None)
            if idx is None:
                raise ValueError(
                    f"derives[{i}].from '{src_key}' is not a key "
                    f"(known: {', '.join(k['name'] for k in keys)})")
            if keys[idx]["vkind"] != "bucket":
                raise ValueError(
                    f"derives[{i}].from '{src_key}' is not a binned key; a key-scope "
                    f"derive needs a bucket to read")
            availability = str(d.get("availability") or "past_only").strip().lower()
            if availability not in AVAIL:
                raise ValueError(
                    f"derive '{name}': availability must be one of {AVAIL}, got '{availability}'")
            features.append({
                "name": name, "out": out, "fn": None, "kind": "derived", "shelf": "core",
                "source": None, "column": "", "entity_col": "",
                "derive": via, "derive_key": idx, "availability": availability,
                "ord": int(d["ord"]) if d.get("ord") is not None else 1000 + i,
            })
        features.sort(key=lambda f: f["ord"])

        if not targets and not features:
            raise ValueError("at least one aggregate is required (nothing would be computed)")

        # -- P2 filters -----------------------------------------------------
        # Always declared here, even when a predicate was pushed down: the SPEC
        # is the reviewed artefact, so a scoping decision that changes the
        # numbers has to stay visible in it. `applied_upstream` marks one the
        # executor should not re-run.
        filters = []
        for i, f in enumerate(p.get("filters") or []):
            if not isinstance(f, dict) or not str(f.get("of") or "").strip():
                raise ValueError(f"filters[{i}] needs a column ('of')")
            op = str(f.get("using") or "equals").strip()
            if op not in FILTER_OPS:
                raise ValueError(f"filters[{i}].using must be one of {FILTER_OPS}, got '{op}'")
            src, col = split(f.get("of"), f"filters[{i}].of")
            filters.append({"source": src, "column": col, "operator": op,
                            "value": f.get("value"),
                            "skip": bool(f.get("applied_upstream"))})

        return {
            "sources": sources, "multi": multi, "source_from": source_from,
            "keys": keys, "spine": spine, "t_key": t_key, "time_grain": time_grain,
            "targets": targets, "features": features, "filters": filters,
        }

    # ---------------------------------------------------------- the recipe

    def _execute(self, rows, spec):
        dropped = {"not_a_row": 0, "unknown_source": 0, "filtered": 0,
                   "bad_time": 0, "bad_measure": 0}
        keys, spine = spec["keys"], spec["spine"]
        src_by_id = {s["id"]: s for s in spec["sources"]}
        tg_by_src, ft_by_src, flt_by_src = {}, {}, {}
        for tg in spec["targets"]:
            tg_by_src.setdefault(tg["source"], []).append(tg)
        for f in spec["features"]:
            if f["source"] is not None:
                ft_by_src.setdefault(f["source"], []).append(f)
        for f in spec["filters"]:
            if not f["skip"]:
                flt_by_src.setdefault(f["source"], []).append(f)

        src_counts = {s["id"]: 0 for s in spec["sources"]}
        buckets = {}
        refiled = 0

        for row in rows:
            if not isinstance(row, dict):
                dropped["not_a_row"] += 1
                continue
            if spec["multi"]:
                src = src_by_id.get(str(row.get(spec["source_from"]) or "").strip())
                if src is None:
                    dropped["unknown_source"] += 1
                    continue
            else:
                src = spec["sources"][0]
            src_counts[src["id"]] += 1
            if not self._passes_filters(row, flt_by_src.get(src["id"], [])):
                dropped["filtered"] += 1
                continue
            # the shared time spine, if this frame has one at all
            ev_lab = ev_sort = av_lab = av_sort = None
            if spine:
                d = self._parse_date(row.get(src["time_from"]))
                if d is None:
                    dropped["bad_time"] += 1
                    continue
                via = keys[spine[0]]["via"]
                ev_lab, ev_sort = derive_library.apply_bucket(via, d)
                # arrival bucket: when this row became knowable (defaults to
                # event time when no available_from is declared, or it's missing)
                av_lab, av_sort = ev_lab, ev_sort
                if src["available_from"]:
                    da = self._parse_date(row.get(src["available_from"]))
                    if da is not None:
                        av_lab, av_sort = derive_library.apply_bucket(via, da)

            built = self._key_tuple(row, keys, src, ev_lab, ev_sort)
            if built is None:
                dropped["bad_time"] += 1
                continue
            klab, ksort = built

            b = buckets.setdefault(klab, {"sort": ksort, "tv": {}, "fv": {}})
            for tg in tg_by_src.get(src["id"], []):
                if tg["kind"] == "group":
                    v = self._parse_number(row.get(tg["column"]))
                    if v is None:
                        dropped["bad_measure"] += 1
                    else:
                        ent = row.get(tg["entity_col"])
                        ent = "" if ent is None else str(ent).strip()
                        d = b["tv"].setdefault(tg["name"], {})
                        d[ent] = d.get(ent, 0.0) + v
                elif tg["fn"] == "count" and not tg["column"]:
                    b["tv"].setdefault(tg["name"], []).append(1.0)
                else:
                    v = self._parse_number(row.get(tg["column"]))
                    if v is None:
                        dropped["bad_measure"] += 1
                    else:
                        b["tv"].setdefault(tg["name"], []).append(v)
            for f in ft_by_src.get(src["id"], []):
                # past_only covariates that arrived after their event bucket are
                # re-filed forward to the bucket where they became knowable, so a
                # value can never leak into a row that predates knowing it.
                forward = (bool(spine) and f["availability"] == "past_only"
                           and av_sort > ev_sort)
                if forward:
                    alab, asort = list(klab), list(ksort)
                    for si in spine:
                        alab[si], asort[si] = av_lab, av_sort
                    fb = buckets.setdefault(tuple(alab),
                                            {"sort": tuple(asort), "tv": {}, "fv": {}})
                else:
                    fb = b
                moved = False
                if f["kind"] == "group":
                    fv = self._parse_number(row.get(f["column"]))
                    if fv is not None:
                        ent = row.get(f["entity_col"])
                        ent = "" if ent is None else str(ent).strip()
                        gd = fb["fv"].setdefault(f["name"], {})
                        gd[ent] = gd.get(ent, 0.0) + fv
                        moved = True
                elif f["fn"] == "count" and not f["column"]:
                    fb["fv"].setdefault(f["name"], []).append(1.0)
                    moved = True
                else:
                    fv = self._parse_number(row.get(f["column"]))
                    if fv is not None:
                        fb["fv"].setdefault(f["name"], []).append(fv)
                        moved = True
                if forward and moved:
                    refiled += 1

        # -- merge (align the sources) & emit ------------------------------
        gaps = {tg["out"]: 0 for tg in spec["targets"]}
        frame = []

        def pick(store, name, b, extras):
            acc = b[store].get(name)
            if acc is None:
                for e in extras:
                    acc = e[store].get(name)
                    if acc is not None:
                        break
            return acc

        def emit(klab, b, extras):
            out = {}
            for i, k in enumerate(keys):
                out[k["name"]] = klab[i]
            # A row is emitted when at least one *measured* aggregate produced a
            # value. Derived columns are always defined, so counting them would
            # emit every bucket; targets alone would drop every clustering row.
            got_any = False
            for tg in spec["targets"]:
                v = self._reduce(tg["fn"], tg["kind"], pick("tv", tg["name"], b, extras))
                out[tg["out"]] = round(v, 6) if v is not None else None
                got_any = got_any or v is not None
            for f in spec["features"]:
                if f["derive"]:
                    out[f["out"]] = derive_library.apply_scalar(
                        f["derive"], b["sort"][f["derive_key"]])
                    continue
                v = self._reduce(f["fn"], f["kind"], pick("fv", f["name"], b, extras))
                out[f["out"]] = round(v, 6) if v is not None else None
                got_any = got_any or v is not None
            if not got_any:
                return None
            for tg in spec["targets"]:
                if out[tg["out"]] is None:
                    gaps[tg["out"]] += 1
            return out

        # A bucket carries _BCAST in every key position its source could not
        # supply; those values broadcast onto each concrete bucket that agrees
        # on the positions they do define.
        concrete = {k: b for k, b in buckets.items() if _BCAST not in k}
        bcast = {k: b for k, b in buckets.items() if _BCAST in k}
        if not concrete and bcast:
            # the owning source never materialized; fold broadcast rows onto an
            # unnamed row rather than silently emitting nothing
            concrete = {tuple("" if v == _BCAST else v for v in k): b
                        for k, b in bcast.items()}
            bcast = {}

        matched_b = set()
        for ck in sorted(concrete, key=lambda k: concrete[k]["sort"]):
            extras = []
            for bk, bb in bcast.items():
                if all(bv == _BCAST or bv == cv for bv, cv in zip(bk, ck)):
                    extras.append(bb)
                    matched_b.add(bk)
            out = emit(ck, concrete[ck], extras)
            if out is not None:
                frame.append(out)
        broadcast_unmatched = len(bcast) - len(matched_b)

        spine_set = set(spine)
        non_spine = [i for i in range(len(keys)) if i not in spine_set]
        meta = {
            "rows_in": len(rows),
            "rows_dropped": dropped,
            "frame_rows": len(frame),
            "series_count": (
                len({tuple(r[keys[i]["name"]] for i in non_spine) for r in frame})
                if (non_spine and frame) else (1 if frame else 0)),
            "t_min": min((r[spec["t_key"]] for r in frame), default=None) if spec["t_key"] else None,
            "t_max": max((r[spec["t_key"]] for r in frame), default=None) if spec["t_key"] else None,
            "grain": spec["time_grain"],
            "keys": [k["name"] for k in keys],
            "targets": [{"name": tg["name"], "fn": tg["fn"], "shelf": tg["shelf"],
                         "source": tg["source"], "measure": tg["column"],
                         "entity": tg["entity_col"] or None} for tg in spec["targets"]],
            "features": [
                {"name": f["name"],
                 "source": f["derive"] or f["column"],
                 "kind": f["kind"],
                 "fn": f["fn"],
                 "availability": f["availability"]}
                for f in spec["features"]
            ],
        }
        if spec["multi"] or len(spec["targets"]) > 1:
            meta["join"] = {"on": "time", "gaps": gaps}
            if spec["multi"]:
                meta["join"]["broadcast_unmatched"] = broadcast_unmatched
                meta["sources"] = {"rows_in": src_counts}
        if any(s["available_from"] for s in spec["sources"]):
            meta["leakage"] = {"features_refiled_forward": refiled}
        return frame, meta

    # -------------------------------------------------------------- helpers

    def _key_tuple(self, row, keys, src, ev_lab, ev_sort):
        """Build one bucket key: (labels, sort values), or None if unparseable.

        A key whose column this row's source does not carry gets the _BCAST
        sentinel, so the row's values broadcast onto every concrete bucket that
        agrees on the positions it does define.
        """
        labs, sorts = [], []
        for k in keys:
            if k["spine"]:
                lab, srt = ev_lab, ev_sort
            elif k["source"] is not None and src["id"] != k["source"]:
                lab = srt = _BCAST
            elif k["vkind"] == "bucket":
                d = self._parse_date(row.get(k["column"]))
                if d is None:
                    return None
                lab, srt = derive_library.apply_bucket(k["via"], d)
            elif k["vkind"] == "scalar":
                d = self._parse_date(row.get(k["column"]))
                if d is None:
                    return None
                lab = srt = derive_library.apply_scalar(k["via"], d)
            else:
                raw = row.get(k["column"])
                raw = "" if raw is None else str(raw).strip()
                lab = srt = k["prefix"] + k["aliases"].get(raw, raw)
            labs.append(lab)
            sorts.append(srt)
        return tuple(labs), tuple(sorts)

    @classmethod
    def _passes_filters(cls, row, filters):
        return all(
            cls._check(row.get(f["column"]), f["operator"], f["value"])
            for f in filters
        )

    @classmethod
    def _check(cls, have, op, want):
        if op == "isEmpty":
            return have is None or str(have).strip() == ""
        if op == "isNotEmpty":
            return not (have is None or str(have).strip() == "")
        if op == "notEquals":
            return not cls._check(have, "equals", want)
        if op == "contains":
            return want is not None and str(want).lower() in str(have or "").lower()
        hn, wn = cls._parse_number(have), cls._parse_number(want)
        if op == "equals":
            if hn is not None and wn is not None:
                return hn == wn
            return str(have) == str(want)
        if hn is None or wn is None:
            return False
        return hn > wn if op == "greaterThan" else hn < wn

    @staticmethod
    def _parse_number(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace(",", "").lstrip("$")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _parse_date(v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        s = str(v).strip()
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            pass
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def _reduce(fn, kind, acc):
        """Collapse one bucket's accumulator to a single value via the library.

        `acc` is a list of values (reduction) or an {entity: net} dict (group);
        empty/None -> None so the emit step can record a join gap.
        """
        if not acc:
            return None
        if kind == "group":
            return function_library.apply_group(fn, acc)
        return MappingLayerService._agg(fn, acc)

    @staticmethod
    def _agg(metric, vals):
        if metric == "count":
            return float(len(vals))
        if not vals:
            return None
        if metric == "sum":
            return float(sum(vals))
        if metric == "mean":
            return float(sum(vals) / len(vals))
        if metric == "min":
            return float(min(vals))
        if metric == "max":
            return float(max(vals))
        if metric == "first":
            return float(vals[0])
        return float(vals[-1])

    # --------------------------------------------------------------- schema

    async def schema(self) -> dict | None:
        SSP = BaseService.ServiceSchemaProperty
        lib = function_library.catalog()
        fn_values = list(lib)
        fn_labels = {n: f"{lib[n]['description']} [{lib[n]['shelf']}]" for n in lib}
        dlib = derive_library.catalog()
        key_derives = [n for n, e in dlib.items() if e["scope"] == "key"]
        derive_labels = {n: dlib[n]["description"] for n in dlib}
        return SSP(
            key="serviceInstructions",
            type="object",
            description=("Mapping-layer SPEC: filter -> derive -> group by keys -> aggregate "
                         "as-of, emitting one declared column per key, aggregate and derive. "
                         "The v1 vocabulary (time_from / series_from / targets / features) "
                         "still runs and is translated into this one."),
            properties=[
                SSP(key="keys", type="array",
                    description=("What identifies one frame row. Rows sharing the same key "
                                 "values are aggregated together; the order here is both the "
                                 "composite-key order and the row sort order"),
                    properties=[
                        SSP(key="name", type="string", required=True,
                            description="Output column name for this key"),
                        SSP(key="from", type="string",
                            description=("Column to read (qualify as <source>.<column> with "
                                         "multiple sources). Omit only on a bin: key, which "
                                         "then bins each source's own event time")),
                        SSP(key="via", type="enum",
                            description=("Derive applied before grouping. A bin: derive turns a "
                                         "date column into a time bucket - this is how a clock "
                                         "becomes an ordinary key instead of a special slot"),
                            enum=ParameterEnum(values=key_derives, labels=derive_labels)),
                        SSP(key="prefix", type="string", default="",
                            description="Optional prefix for this key's values (e.g. 'store_')"),
                        SSP(key="aliases", type="array",
                            description="Entity resolution: rewrite raw key values before grouping",
                            properties=[
                                SSP(key="value", type="string", required=True),
                                SSP(key="replace_with", type="string", required=True),
                            ]),
                    ]),
                SSP(key="aggregates", type="array",
                    description=("One entry per output column that is not a key. Role is a "
                                 "field, so a frame may carry several targets or none at all"),
                    properties=[
                        SSP(key="name", type="string", required=True,
                            description="Output column name"),
                        SSP(key="using", type="enum", required=True,
                            description="Function from the library, applied within each key bucket",
                            enum=ParameterEnum(values=fn_values, labels=fn_labels)),
                        SSP(key="of", type="string",
                            description=("Measure column the function reads; not needed for count")),
                        SSP(key="by", type="string",
                            description=("Entity column to form shares over - required for group "
                                         "functions like hhi and top_share")),
                        SSP(key="role", type="enum", default="target",
                            description="target = predict it; feature = carry it alongside",
                            enum=ParameterEnum(values=list(ROLES))),
                        SSP(key="availability", type="enum", default="past_only",
                            description="Features only: is this knowable at prediction time?",
                            enum=ParameterEnum(
                                values=list(AVAIL),
                                labels={"known_ahead": "Known ahead (safe covariate)",
                                        "past_only": "Past only (history-derived)"})),
                        SSP(key="source", type="string",
                            description="Source id (only needed for a count with multiple sources)"),
                    ]),
                SSP(key="derives", type="array",
                    description="Calendar columns computed from a binned key, not from the rows",
                    properties=[
                        SSP(key="name", type="string", required=True,
                            description="Output column name"),
                        SSP(key="scope", type="enum", default="key",
                            description="key = once per output row (row scope is not built yet)",
                            enum=ParameterEnum(values=["key", "row"])),
                        SSP(key="from", type="string",
                            description="Name of the binned key to read (defaults to the time key)"),
                        SSP(key="via", type="enum", required=True,
                            enum=ParameterEnum(values=key_derives, labels=derive_labels)),
                    ]),
                SSP(key="validity", type="object",
                    description=("The leakage guard. Declares when a row became knowable, as "
                                 "distinct from when it happened"),
                    properties=[
                        SSP(key="arrival_from", type="string",
                            description=("Arrival column. past_only aggregates from rows that "
                                         "arrived after their event bucket are re-filed forward "
                                         "to the bucket where they became knowable, so a "
                                         "covariate never leaks into an earlier row")),
                    ]),
                SSP(key="sources", type="array",
                    description=("Raw sources feeding this frame. Leave empty for a single "
                                 "unnamed source (then top-level time_from applies)"),
                    properties=[
                        SSP(key="id", type="string", required=True,
                            description="Source id, used to qualify columns as <source>.<column>"),
                        SSP(key="time_from", type="string",
                            description="This source's timestamp column (defaults to top-level time_from)"),
                        SSP(key="available_from", type="string",
                            description=("Optional arrival column: when each row became knowable. "
                                         "past_only features from rows that arrived after their event "
                                         "bucket are re-filed forward to the bucket where they became "
                                         "knowable, so a covariate never leaks into an earlier row")),
                    ]),
                SSP(key="source_from", type="string",
                    description=("Column tagging each raw row with its source id "
                                 "(required when more than one source is declared)"),
                    conditional={
                        "enabled": True,
                        "conditions": [{"field": "sources", "operator": "isNotEmpty",
                                        "value": None, "action": "show"}],
                        "logic": "all",
                    }),
                SSP(key="time_from", type="string",
                    description="Timestamp column (single source, or default for sources without their own)"),
                SSP(key="time_grain", type="enum", default="week",
                    description=("v1: time bucket for t. In the keys vocabulary this is a "
                                 "bin: derive on a key, with no default"),
                    enum=ParameterEnum(
                        values=list(GRAINS),
                        labels={"day": "Daily", "week": "Weekly (ISO)", "month": "Monthly",
                                "quarter": "Quarterly", "year": "Yearly"})),
                SSP(key="series_from", type="string",
                    description=("v1: column identifying the series, emitted as series_id. "
                                 "Superseded by keys[], which allows more than one")),
                SSP(key="series_prefix", type="string", default="",
                    description="v1: prefix for series_id values (e.g. 'store_' turns 12 into store_12)"),
                SSP(key="series_aliases", type="array",
                    description="v1: rewrite raw series values before grouping (e.g. merge subsidiaries)",
                    properties=[
                        SSP(key="value", type="string", required=True,
                            description="Raw value as it appears in the data"),
                        SSP(key="replace_with", type="string", required=True,
                            description="Canonical value it becomes"),
                    ]),
                SSP(key="targets", type="array",
                    description=("v1: what to predict; a single target defaults to column 'y'. "
                                 "Superseded by aggregates[] with role=target"),
                    properties=[
                        SSP(key="name", type="string",
                            description="Output column name (e.g. y_power); required with multiple targets"),
                        SSP(key="fn", type="enum", default="sum",
                            description=("Function from the library used to build this target within each "
                                         "(series, t) bucket. Core functions reduce a measure; domain "
                                         "functions like hhi also need an 'entity' column"),
                            enum=ParameterEnum(values=fn_values, labels=fn_labels)),
                        SSP(key="measure", type="string",
                            description=("Column the function reads; qualify as <source>.<column> with "
                                         "multiple sources. Not needed for count")),
                        SSP(key="entity", type="string",
                            description=("Entity column to form shares over — required for group/domain "
                                         "functions like hhi and top_share (e.g. the vendor column)")),
                        SSP(key="source", type="string",
                            description="Source id (only needed for a count target with multiple sources)"),
                    ]),
                SSP(key="filters", type="array",
                    description=("Keep only rows that pass every condition; with multiple sources a "
                                 "qualified column applies only to that source's rows. Declared "
                                 "here even when pushed down, so the SPEC still describes what "
                                 "was computed"),
                    properties=[
                        SSP(key="of", type="string", required=True,
                            description="Raw column to test (qualify as <source>.<column> with multiple sources)"),
                        SSP(key="using", type="enum", default="equals",
                            enum=ParameterEnum(values=list(FILTER_OPS))),
                        SSP(key="value", type="string",
                            description="Comparison value (unused for isEmpty/isNotEmpty)"),
                        SSP(key="applied_upstream", type="boolean", default=False,
                            description=("This predicate already ran upstream: keep it declared "
                                         "for provenance, but do not re-run it here")),
                    ]),
                SSP(key="features", type="array",
                    description=("v1: extra x_<name> columns carried alongside the targets. "
                                 "Superseded by aggregates[] with role=feature, plus derives[]"),
                    properties=[
                        SSP(key="name", type="string", required=True,
                            description="Feature name; emitted as column x_<name>"),
                        SSP(key="measure", type="string",
                            description=("Source column the function reads (qualify as <source>.<column> "
                                         "with multiple sources; leave empty for a derived feature)")),
                        SSP(key="fn", type="enum", default="max",
                            description="Function from the library used to build this feature within the bucket",
                            enum=ParameterEnum(values=fn_values, labels=fn_labels)),
                        SSP(key="entity", type="string",
                            description="Entity column — required for group/domain functions like hhi"),
                        SSP(key="derive", type="enum",
                            description="Calendar-derived feature computed from the bucket (instead of a measure)",
                            enum=ParameterEnum(values=list(DERIVES))),
                        SSP(key="availability", type="enum", default="past_only",
                            description="Leakage flag consumed by the fitness gates: is this knowable at forecast time?",
                            enum=ParameterEnum(
                                values=["known_ahead", "past_only"],
                                labels={"known_ahead": "Known ahead (safe covariate)",
                                        "past_only": "Past only (history-derived)"})),
                    ]),
            ],
        )

    # ------------------------------------------------------------ self-test

    def self_test(self):
        """Offline check against the flowchart's own samples: retail single-source
        and the power+weather multi-source join."""
        checks = {}

        # ---- single source: retail weekly sales (v1-compat flat spec) ----
        retail = [
            {"date": "2024-01-03", "store_nbr": "12", "family": "BEVERAGES", "sales": "402.10", "onpromotion": "1"},
            {"date": "2024-01-03", "store_nbr": "12", "family": "BREAD", "sales": "88.00", "onpromotion": "0"},
            {"date": "2024-01-04", "store_nbr": "12", "family": "BEVERAGES", "sales": "377.50", "onpromotion": "1"},
            {"date": "2024-01-08", "store_nbr": "12", "family": "BEVERAGES", "sales": "410.00", "onpromotion": "0"},
            {"date": "2024-01-03", "store_nbr": "44", "family": "BEVERAGES", "sales": "903.20", "onpromotion": "0"},
            {"date": "not-a-date", "store_nbr": "12", "family": "BREAD", "sales": "1.00", "onpromotion": "0"},
        ]
        flat = {
            "series_from": "store_nbr", "series_prefix": "store_",
            "time_from": "date", "time_grain": "week",
            "target_metric": "sum", "target_measure": "sales",
            "features": [
                {"name": "promo", "measure": "onpromotion", "metric": "max", "availability": "known_ahead"},
                {"name": "week_of_year", "derive": "week_of_year", "availability": "known_ahead"},
            ],
        }
        frame, meta = self._execute(retail, self._resolve_spec(flat))

        def cell(fr, sid, t):
            return next((r for r in fr if r.get("series_id") == sid and r["t"] == t), None)

        checks["v1 flat spec still runs"] = len(frame) == 3 and meta["series_count"] == 2
        checks["single target lands in column y"] = abs(cell(frame, "store_12", "2024-W01")["y"] - 867.6) < 1e-6
        checks["promo aggregated with max"] = cell(frame, "store_12", "2024-W01")["x_promo"] == 1.0
        checks["derived week_of_year"] = cell(frame, "store_12", "2024-W01")["x_week_of_year"] == 1
        checks["bad date dropped and counted"] = meta["rows_dropped"]["bad_time"] == 1
        checks["availability reaches metadata"] = meta["features"][0]["availability"] == "known_ahead"
        checks["no join block for single target"] = "join" not in meta

        # same recipe expressed as targets[]
        spec_t = self._resolve_spec({
            "series_from": "store_nbr", "series_prefix": "store_",
            "time_from": "date", "time_grain": "week",
            "targets": [{"metric": "sum", "measure": "sales"}],
        })
        frame_t, _ = self._execute(retail, spec_t)
        checks["targets[] form matches flat form"] = (
            abs(cell(frame_t, "store_12", "2024-W01")["y"] - 867.6) < 1e-6)

        spec_f = self._resolve_spec({**flat, "filters": [
            {"column": "family", "operator": "equals", "value": "BEVERAGES"}]})
        frame_f, meta_f = self._execute(retail, spec_f)
        checks["filter drops non-matching rows"] = (
            abs(cell(frame_f, "store_12", "2024-W01")["y"] - 779.6) < 1e-6
            and meta_f["rows_dropped"]["filtered"] == 2)

        spec_a = self._resolve_spec({**flat, "series_aliases": [
            {"value": "44", "replace_with": "12"}]})
        frame_a, _ = self._execute(retail, spec_a)
        checks["aliases merge series"] = abs(cell(frame_a, "store_12", "2024-W01")["y"] - 1770.8) < 1e-6

        # ---- multiple sources: power + weather, joined on time ----------
        pw = [
            {"src": "power",   "period": "2024-07-01 13:00", "value_MWh": "40,910"},
            {"src": "power",   "period": "2024-07-01 14:00", "value_MWh": "41,530"},
            {"src": "power",   "period": "2024-07-02 13:00", "value_MWh": "43,850"},
            {"src": "weather", "time": "2024-07-01 13:00", "temperature_c": "28.1"},
            {"src": "weather", "time": "2024-07-01 14:00", "temperature_c": "28.9"},
            {"src": "weather", "time": "2024-07-03 13:00", "temperature_c": "24.8"},
            {"src": "solar",   "time": "2024-07-01 13:00", "kw": "5"},
        ]
        multi = {
            "source_from": "src",
            "sources": [{"id": "power", "time_from": "period"},
                        {"id": "weather", "time_from": "time"}],
            "time_grain": "day",
            "targets": [
                {"name": "y_power", "metric": "mean", "measure": "power.value_MWh"},
                {"name": "y_temp", "metric": "mean", "measure": "weather.temperature_c"},
            ],
        }
        frame_m, meta_m = self._execute(pw, self._resolve_spec(multi))

        def day(fr, t):
            return next((r for r in fr if r["t"] == t), None)

        checks["joined frame has no series_id"] = frame_m and "series_id" not in frame_m[0]
        checks["sources merged on shared t"] = (
            abs(day(frame_m, "2024-07-01")["y_power"] - 41220.0) < 1e-6
            and abs(day(frame_m, "2024-07-01")["y_temp"] - 28.5) < 1e-6)
        checks["join gap emits null not a dropped row"] = (
            day(frame_m, "2024-07-02")["y_temp"] is None
            and day(frame_m, "2024-07-03")["y_power"] is None)
        checks["join gaps counted per target"] = (
            meta_m["join"]["gaps"] == {"y_power": 1, "y_temp": 1})
        checks["undeclared source dropped and counted"] = (
            meta_m["rows_dropped"]["unknown_source"] == 1)
        checks["per-source row counts in metadata"] = (
            meta_m["sources"]["rows_in"] == {"power": 3, "weather": 3})

        # series on one source broadcasts the other onto every series
        pw_series = [
            {"src": "power", "period": "2024-07-01", "respondent": "PJM", "value_MWh": "40000"},
            {"src": "power", "period": "2024-07-01", "respondent": "MISO", "value_MWh": "30000"},
            {"src": "weather", "time": "2024-07-01", "temperature_c": "28.5"},
        ]
        spec_b = self._resolve_spec({**multi, "series_from": "power.respondent"})
        frame_b, _ = self._execute(pw_series, spec_b)
        checks["seriesless source broadcast to every series"] = (
            cell(frame_b, "PJM", "2024-07-01")["y_temp"] == 28.5
            and cell(frame_b, "MISO", "2024-07-01")["y_temp"] == 28.5
            and cell(frame_b, "MISO", "2024-07-01")["y_power"] == 30000.0)

        # qualified filter touches only its own source
        spec_qf = self._resolve_spec({**multi, "filters": [
            {"column": "power.value_MWh", "operator": "greaterThan", "value": "41000"}]})
        frame_qf, _ = self._execute(pw, spec_qf)
        checks["qualified filter scoped to its source"] = (
            abs(day(frame_qf, "2024-07-01")["y_power"] - 41530.0) < 1e-6
            and abs(day(frame_qf, "2024-07-01")["y_temp"] - 28.5) < 1e-6)

        # ---- spec validation --------------------------------------------
        def rejects(params):
            try:
                self._resolve_spec(params)
                return False
            except ValueError:
                return True

        checks["missing time rejected"] = rejects({"targets": [{"measure": "sales"}]})
        checks["multi-source needs source_from"] = rejects(
            {**multi, "source_from": ""})
        checks["multi-source unqualified measure rejected"] = rejects(
            {**multi, "targets": [{"name": "y_power", "measure": "value_MWh"}]})
        checks["duplicate target names rejected"] = rejects(
            {**multi, "targets": [{"name": "y", "measure": "power.value_MWh"},
                                  {"name": "y", "measure": "weather.temperature_c"}]})
        checks["unnamed second target rejected"] = rejects(
            {**multi, "targets": [{"measure": "power.value_MWh"},
                                  {"measure": "weather.temperature_c"}]})
        checks["target/source mismatch rejected"] = rejects(
            {**multi, "targets": [{"name": "y", "source": "weather", "measure": "power.value_MWh"}]})

        # ---- function library: group functions (hhi, top_share) ----------
        # Raw award rows -> HHI per category per year, straight from the library.
        # Pins to the concentration spec's worked examples (60/30/10 -> 4600,
        # twenty-at-5% -> 500, deobligation nets 50/50 -> 5000).
        awards = (
            [{"cat": "A", "yr": "2024-01-01", "vendor": "v1", "amt": "60000000"},
             {"cat": "A", "yr": "2024-01-01", "vendor": "v2", "amt": "30000000"},
             {"cat": "A", "yr": "2024-01-01", "vendor": "v3", "amt": "10000000"}]
            + [{"cat": "B", "yr": "2024-01-01", "vendor": f"v{i}", "amt": "5000000"} for i in range(20)]
            + [{"cat": "C", "yr": "2024-01-01", "vendor": "v1", "amt": "100000000"},
               {"cat": "C", "yr": "2024-01-01", "vendor": "v1", "amt": "-40000000"},
               {"cat": "C", "yr": "2024-01-01", "vendor": "v2", "amt": "60000000"}]
        )
        hhi_spec = self._resolve_spec({
            "series_from": "cat", "time_from": "yr", "time_grain": "year",
            "targets": [{"name": "y", "fn": "hhi", "measure": "amt", "entity": "vendor"}],
            "features": [{"name": "top", "fn": "top_share", "measure": "amt",
                          "entity": "vendor", "availability": "past_only"}],
        })
        hhi_frame, hhi_meta = self._execute(awards, hhi_spec)
        checks["hhi computed from raw rows (concentrated 4600)"] = (
            abs(cell(hhi_frame, "A", "2024")["y"] - 4600.0) < 1e-6)
        checks["hhi computed from raw rows (spread 500)"] = (
            abs(cell(hhi_frame, "B", "2024")["y"] - 500.0) < 1e-6)
        checks["hhi nets deobligations per entity (50/50 -> 5000)"] = (
            abs(cell(hhi_frame, "C", "2024")["y"] - 5000.0) < 1e-6)
        checks["top_share as a group feature"] = (
            abs(cell(hhi_frame, "A", "2024")["x_top"] - 60.0) < 1e-6)
        checks["group function reaches metadata with its shelf"] = (
            hhi_meta["targets"][0]["fn"] == "hhi"
            and hhi_meta["targets"][0]["shelf"] == "supply-chain"
            and hhi_meta["targets"][0]["entity"] == "vendor")

        # non-positive category base -> undefined -> null -> row omitted
        neg = [{"cat": "D", "yr": "2024-01-01", "vendor": "v1", "amt": "-10"},
               {"cat": "D", "yr": "2024-01-01", "vendor": "v2", "amt": "5"}]
        neg_frame, _ = self._execute(neg, hhi_spec)
        checks["hhi undefined on non-positive base (no row)"] = neg_frame == []

        checks["hhi needs an entity column"] = rejects(
            {"series_from": "cat", "time_from": "yr", "time_grain": "year",
             "targets": [{"name": "y", "fn": "hhi", "measure": "amt"}]})
        checks["unknown function rejected"] = rejects(
            {"series_from": "cat", "time_from": "yr", "time_grain": "year",
             "targets": [{"name": "y", "fn": "gini", "measure": "amt", "entity": "vendor"}]})

        # the catalog .csv (if shipped) must match the embedded fallback
        checks["csv catalog matches embedded fallback"] = (
            function_library.catalog() == function_library.embedded_catalog())

        # ---- availability: late covariate re-filed forward (leakage guard) ---
        # A promo dated 2024-07-20 but not knowable until 2024-09-05 must NOT
        # land in July's row (that would be leakage) — it belongs in September,
        # the bucket where a forecaster would first have had it.
        late = [
            {"day": "2024-07-15", "known": "2024-07-15", "sales": "100", "promo": "0"},
            {"day": "2024-08-15", "known": "2024-08-15", "sales": "100", "promo": "0"},
            {"day": "2024-09-15", "known": "2024-09-15", "sales": "100", "promo": "0"},
            {"day": "2024-07-20", "known": "2024-09-05", "sales": "0", "promo": "1"},  # late
        ]
        late_spec = self._resolve_spec({
            "time_grain": "month",
            "sources": [{"id": "s", "time_from": "day", "available_from": "known"}],
            "targets": [{"name": "y", "fn": "sum", "measure": "sales"}],
            "features": [{"name": "promo", "fn": "max", "measure": "promo",
                          "availability": "past_only"}],
        })
        late_frame, late_meta = self._execute(late, late_spec)

        def mon(fr, t):
            return next((r for r in fr if r["t"] == t), None)

        checks["late covariate kept out of its event bucket"] = (
            mon(late_frame, "2024-07")["x_promo"] == 0.0)
        checks["late covariate re-filed to arrival bucket"] = (
            mon(late_frame, "2024-09")["x_promo"] == 1.0)
        checks["re-file counted in metadata"] = (
            late_meta["leakage"]["features_refiled_forward"] == 1)

        # control: a known_ahead covariate is NOT re-filed (safe by definition),
        # so the same promo lands in July and nothing moves.
        ka_spec = self._resolve_spec({
            "time_grain": "month",
            "sources": [{"id": "s", "time_from": "day", "available_from": "known"}],
            "targets": [{"name": "y", "fn": "sum", "measure": "sales"}],
            "features": [{"name": "promo", "fn": "max", "measure": "promo",
                          "availability": "known_ahead"}],
        })
        ka_frame, ka_meta = self._execute(late, ka_spec)
        checks["known_ahead covariate not re-filed"] = (
            mon(ka_frame, "2024-07")["x_promo"] == 1.0
            and ka_meta["leakage"]["features_refiled_forward"] == 0)

        # no available_from declared -> no leakage block, behaviour unchanged
        plain_spec = self._resolve_spec({
            "time_from": "day", "time_grain": "month",
            "targets": [{"name": "y", "fn": "sum", "measure": "sales"}],
            "features": [{"name": "promo", "fn": "max", "measure": "promo",
                          "availability": "past_only"}],
        })
        _, plain_meta = self._execute(late, plain_spec)
        checks["no leakage block without available_from"] = "leakage" not in plain_meta

        # ---- v2: positioned vocabulary -----------------------------------
        # Same retail rows, same numbers, but every column is declared. v1 is
        # now just the dialect the translation writes; nothing above changed.
        v2 = {
            "time_from": "date",
            "keys": [{"name": "store", "from": "store_nbr", "prefix": "store_"},
                     {"name": "period", "via": "bin:week"}],
            "aggregates": [
                {"name": "revenue", "using": "sum", "of": "sales", "role": "target"},
                {"name": "promo", "using": "max", "of": "onpromotion",
                 "role": "feature", "availability": "known_ahead"},
            ],
            "derives": [{"name": "wk", "scope": "key", "from": "period",
                         "via": "week_of_year"}],
        }
        f2, m2 = self._execute(retail, self._resolve_spec(v2))
        row = next(r for r in f2 if r["store"] == "store_12" and r["period"] == "2024-W01")
        checks["v2 emits declared column names"] = (
            set(row) == {"store", "period", "revenue", "promo", "wk"})
        checks["v2 matches the v1 numbers exactly"] = (
            abs(row["revenue"] - 867.6) < 1e-6 and row["promo"] == 1.0 and row["wk"] == 1)
        checks["v2 reports its keys"] = m2["keys"] == ["store", "period"]
        checks["v2 series_count over non-time keys"] = m2["series_count"] == 2

        # clustering: no target anywhere. Every bucket is emitted on the
        # strength of its features alone - the case the old got_any gate ate.
        clu = self._resolve_spec({
            "keys": [{"name": "customer", "from": "store_nbr"}],
            "aggregates": [
                {"name": "spend", "using": "sum", "of": "sales", "role": "feature"},
                {"name": "orders", "using": "count", "role": "feature"}],
        })
        f_clu, m_clu = self._execute(retail, clu)
        checks["clustering: no target, rows still emitted"] = (
            len(f_clu) == 2
            # 1277.6 from the four parseable rows, plus the 1.00 whose date is
            # junk: with no clock nothing parses it, so it legitimately counts
            and abs(next(r for r in f_clu if r["customer"] == "12")["spend"] - 1278.6) < 1e-6)
        checks["clustering: unparseable date is irrelevant without a clock"] = (
            m_clu["rows_dropped"]["bad_time"] == 0)
        checks["clustering: no clock, no grain, no t"] = (
            m_clu["grain"] == "" and m_clu["t_min"] is None)

        # recommendation: the row identity is a pair, not a series
        rec = self._resolve_spec({
            "keys": [{"name": "store", "from": "store_nbr"},
                     {"name": "item", "from": "family"}],
            "aggregates": [{"name": "rating", "using": "mean", "of": "sales",
                            "role": "target"}],
        })
        f_rec, _ = self._execute(retail, rec)
        checks["recommendation: two key columns"] = (
            len(f_rec) == 3
            and all(set(r) == {"store", "item", "rating"} for r in f_rec))

        # a filter pushed down upstream stays declared but is not re-run
        push = self._resolve_spec({**v2, "filters": [
            {"of": "family", "using": "equals", "value": "BEVERAGES",
             "applied_upstream": True}]})
        f_push, m_push = self._execute(retail, push)
        checks["applied_upstream filter declared, not executed"] = (
            len(push["filters"]) == 1 and push["filters"][0]["skip"]
            and m_push["rows_dropped"]["filtered"] == 0
            and abs(next(r for r in f_push
                         if r["store"] == "store_12" and r["period"] == "2024-W01"
                         )["revenue"] - 867.6) < 1e-6)

        # three-state resolution reaches the derive library too
        checks["unknown derive rejected"] = rejects({**v2, "derives": [
            {"name": "d", "scope": "key", "from": "period", "via": "fortnight_of_year"}]})
        checks["unknown grain rejected, never defaulted"] = rejects({
            "time_from": "date",
            "keys": [{"name": "period", "via": "bin:fortnight"}],
            "aggregates": [{"name": "y", "using": "sum", "of": "sales", "role": "target"}]})
        checks["aggregate colliding with a key rejected"] = rejects({**v2, "aggregates": [
            {"name": "store", "using": "sum", "of": "sales", "role": "target"}]})
        checks["derive off a non-binned key rejected"] = rejects({**v2, "derives": [
            {"name": "d", "scope": "key", "from": "store", "via": "week_of_year"}]})
        checks["spec with no aggregate rejected"] = rejects({
            "time_from": "date", "keys": [{"name": "period", "via": "bin:week"}]})
        checks["derive csv catalog matches embedded fallback"] = (
            derive_library.catalog() == derive_library.embedded_catalog())

        return {"pass": all(checks.values()), "checks": checks}
