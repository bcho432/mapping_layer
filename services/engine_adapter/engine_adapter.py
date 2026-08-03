"""engine-adapter: the canonical frame -> one engine's dialect, and back.

The fourth service, and the first thing downstream of the frame. It closes the
loop the bench has been missing:

    frame + run config  ->  engine-adapter  ->  predictions + accuracy

Three jobs, and it is worth being clear that they are separate:

  1. DIALECT. Every engine names the same three ideas differently. Nixtla wants
     `unique_id / ds / y`; AutoGluon wants `item_id / timestamp / target` plus
     `known_covariates`. The frame speaks neither. Renaming is all that stands
     between them, and doing it here is what keeps the mapping layer from
     having to know which library it is feeding.

  2. THE FUTURE ROWS. A forecast needs somewhere to put its answer. The horizon
     and the frame's own clock are enough to scaffold those rows, and the
     covariates that are `known_ahead` can be carried into them; the ones that
     are `past_only` cannot, and leaving them null is the honest thing.

  3. COMPATIBILITY. Season length against the grain, horizon against whether
     there is a clock at all, engine against the number of entity keys. The
     bench checked these in the browser, where it could only read the
     documents. Here they can also read the data.

WHY THE ROLE SPLIT MATTERS HERE
-------------------------------
The SPEC no longer carries `role`; the run config names the target. That is
what makes this service possible without recompiling: point `target` at a
different column of the same frame and the frame does not move — only what this
adapter feeds the model does.

RUNNING A MODEL
---------------
`engine: local` fits stdlib exponential smoothing (see ets.py) and returns real
predictions with a rolling-origin accuracy estimate, so the loop closes with no
external dependency. Any other engine is prepared but not run: the dialect
payload comes back for you to POST wherever that engine lives.

Stdlib only - no package_dependencies.
"""

import asyncio

from spl.core.base_service.base_service_class import BaseService
from spl.core.service_types import ParameterEnum

try:  # sibling modules: package-relative when deployed, flat when run locally
    from . import classify, ets, recommend
except ImportError:  # pragma: no cover
    import classify
    import ets
    import recommend

ENGINES = ("local", "nixtla", "autogluon", "recbole")
MODES = ("run", "prepare")

# Which knobs each task actually READS. A forecast has a horizon and a season;
# a recommendation has neither and has a top_k instead. Declaring it here and
# letting the UI render from the declaration is the same rule the SPEC follows:
# a knob nobody reads is a typo or a lie, and showing one is worse than both
# because it looks like it did something.
TASK_KNOBS = {
    "forecast":  ["target", "features", "model", "horizon", "season_length",
                  "quantiles", "backtest"],
    "recommend": ["target", "top_k", "holdout", "folds",
                  "user_key", "item_key"],
    "classify":  ["target", "features", "model", "folds", "trees", "depth",
                  "class_weight", "positive_class", "threshold"],
}

# grain -> how many buckets make one cycle. A season length that disagrees with
# the grain is the single most common way a forecast is quietly wrong: fitting
# 12 on quarterly data asks for a three-year cycle nobody meant.
NATURAL_SEASON = {"day": 7, "week": 52, "month": 12, "quarter": 4, "year": 1}

# The dialect each engine speaks. Series id, clock, target.
DIALECT = {
    "nixtla":    {"series": "unique_id", "clock": "ds", "target": "y",
                  "exog": "X_df"},
    "autogluon": {"series": "item_id", "clock": "timestamp", "target": "target",
                  "exog": "known_covariates"},
    "local":     {"series": "series_id", "clock": "t", "target": "y",
                  "exog": None},
    # RecBole reads atomic files whose headers carry the field type. There is
    # no clock slot: an interaction file is a user x item grid, and time is an
    # optional extra column rather than the axis.
    "recbole":   {"series": "user_id:token", "clock": "timestamp:float",
                  "target": "rating:float", "item": "item_id:token",
                  "exog": None},
}


class EngineAdapterService(BaseService):

    # ------------------------------------------------------------------ run

    async def _run(self, request: BaseService.RunRequest) -> BaseService.RunResponse:
        frame = request.data or []
        p = self._param_root(request.parameters or {})
        try:
            out, report = await asyncio.to_thread(self._adapt, frame, p)
        except ValueError as e:
            self.logger.error(f"engine-adapter: {e}")
            return BaseService.RunResponse(data=[{"error": str(e)}])

        for w in report.get("warnings", []):
            self.logger.warning(f"engine-adapter: {w}")
        self.logger.info(
            f"engine-adapter: {report['engine']} ({report['mode']}) over "
            f"{report['series_count']} series, horizon {report['horizon']}")
        return BaseService.RunResponse(data=out, metadata=report)

    @staticmethod
    def _param_root(parameters):
        inner = parameters.get("serviceInstructions")
        return inner if isinstance(inner, dict) else parameters

    # -------------------------------------------------------------- adapt

    def _adapt(self, frame, p):
        if not frame:
            raise ValueError("no frame: the mapping layer produced no rows")

        rc = p.get("run_config") or {}
        meta = p.get("frame_meta") or {}
        engine = str(rc.get("engine") or "local").strip().lower()
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got '{engine}'")
        mode = str(p.get("mode") or ("run" if engine == "local" else "prepare")).lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")

        cols = list(frame[0].keys())
        clock, series_keys = self._axes(meta, cols)
        target = self._target(rc, meta, cols, clock, series_keys)
        features = [c for c in (rc.get("features") or []) if c in cols]

        horizon = int(rc.get("horizon") or 0)
        season = int(rc.get("season_length") or 0) or None
        grain = meta.get("grain")

        kind = str(rc.get("kind") or "").strip().lower()
        warnings, checks = self._compat(engine, clock, series_keys, horizon,
                                        season, grain, frame, target, kind)

        report = {
            "engine": engine, "mode": mode,
            "dialect": DIALECT[engine],
            "target": target, "features": features,
            "clock": clock, "series_keys": series_keys,
            "grain": grain, "horizon": horizon, "season_length": season,
            "series_count": self._series_count(frame, series_keys),
            "checks": checks, "warnings": warnings,
        }

        payload = self._to_dialect(frame, engine, clock, series_keys, target,
                                   features, meta)
        report["payload_rows"] = len(payload)

        if mode == "prepare" or engine != "local":
            report["note"] = (
                f"prepared only — POST `data` to {engine}. Nothing was fitted "
                f"here; set engine 'local' to fit stdlib exponential smoothing.")
            return payload, report

        # The profile already said what question this is, and the compiler put
        # it in the run config. Reading it is more honest than re-deriving it
        # from the frame's shape — a classification and a clustering frame look
        # identical apart from a column, and guessing between them is exactly
        # the kind of inference this pipeline exists to avoid.
        if str(rc.get("kind") or "").strip().lower() == "classify" and not clock:
            preds, accuracy = self._run_classify(frame, series_keys, target,
                                                 features, rc, meta)
            report["task"] = "classify"
            report["knobs"] = TASK_KNOBS["classify"]
            report["ignored_knobs"] = sorted(
                k for k in rc
                if k not in TASK_KNOBS["classify"] + ["engine", "kind"]
                and rc[k] not in (None, "", [], {}))
            report["accuracy"] = accuracy
            report["verdict"] = classify.verdict(accuracy)
            return preds, report

        # A grid of two entity keys and no clock is a recommendation, not a
        # forecast. The frame says which of the two it is; nothing else has to.
        if len(series_keys) >= 2 and not clock:
            preds, accuracy = self._run_recommend(frame, series_keys, target, rc)
            report["task"] = "recommend"
            report["knobs"] = TASK_KNOBS["recommend"]
            report["ignored_knobs"] = sorted(
                k for k in rc
                if k not in TASK_KNOBS["recommend"] + ["engine"] and rc[k]
                not in (None, "", [], {}))
            report["accuracy"] = accuracy
            report["verdict"] = recommend.verdict(accuracy)
            return preds, report

        preds, accuracy = self._run_local(frame, clock, series_keys, target,
                                          horizon, season)
        report["task"] = "forecast"
        report["knobs"] = TASK_KNOBS["forecast"]
        report["ignored_knobs"] = sorted(
            k for k in rc
            if k not in TASK_KNOBS["forecast"] + ["engine"] and rc[k]
            not in (None, "", [], {}))
        report["accuracy"] = accuracy
        report["verdict"] = self._verdict(accuracy)
        return preds, report

    # ---------------------------------------------------- local classifier

    def _run_classify(self, frame, series_keys, target, features, rc, meta):
        """Random forest over the frame's feature columns, scored by k-fold.

        The frame carries integer class ids, because it is a numeric matrix by
        construction; `frame_meta.classes` carries the names. Predictions are
        mapped back to names on the way out, so the engine works in ids and the
        reader sees labels.
        """
        cols = list(frame[0].keys())
        feats = [c for c in (features or [])
                 if c in cols and c != target and c not in series_keys]
        if not feats:
            feats = [c for c in cols if c != target and c not in series_keys]
        if not feats:
            raise ValueError("no feature columns: a classifier needs something "
                             "to learn from besides the label")

        names = ((meta.get("classes") or {}).get(target) or {})
        label_of = (lambda v: names.get(str(int(v)), v)) if names else (lambda v: v)

        # Three populations, and conflating them is why unlabelled rows used to
        # come back with a null prediction:
        #   scorable   a label AND complete features -> trains and is scored
        #   to_predict no label, complete features   -> THE DELIVERABLE
        #   incomplete a feature is missing          -> cannot be predicted at all
        rows, labels, keep = [], [], []
        to_predict, incomplete = [], []
        for i, r in enumerate(frame):
            y = r.get(target)
            xs = [r.get(f) for f in feats]
            if any(not _isnum(x) for x in xs):
                incomplete.append(i)
            elif y is None:
                to_predict.append(i)
            else:
                rows.append([float(x) for x in xs])
                labels.append(label_of(y))
                keep.append(i)
        if len(rows) < 4:
            raise ValueError(
                f"only {len(rows)} rows have a label and complete features — "
                f"too few to fit and score a classifier")

        positive = str(rc.get("positive_class") or "").strip() or None
        if positive is not None and positive not in set(labels):
            raise ValueError(
                f"positive_class {positive!r} is not one of the labels "
                f"({', '.join(sorted(set(labels)))})")

        res, why = classify.evaluate(
            rows, labels,
            k=int(rc.get("folds") or 5),
            n_trees=int(rc.get("trees") or classify.DEFAULT_TREES),
            depth=int(rc.get("depth") or classify.DEFAULT_DEPTH),
            class_weight=(rc.get("class_weight") or None),
            positive=positive,
            threshold=float(rc.get("threshold") or 0.5))
        if res is None:
            raise ValueError(why)

        acc = classify.score(labels, res["pred"], res["proba"], positive=positive)
        acc.update({"folds": res["folds"], "features": feats,
                    "model": rc.get("model") or "random_forest",
                    "threshold": float(rc.get("threshold") or 0.5),
                    "class_weight": rc.get("class_weight") or "none"})
        if to_predict:
            acc["rows_predicted"] = len(to_predict)
        if incomplete:
            acc["rows_unscorable"] = len(incomplete)
            acc["unscorable_reason"] = (
                f"a feature was missing ({', '.join(feats[:3])}"
                f"{'...' if len(feats) > 3 else ''})")

        # The rows with no known outcome are the point of the exercise, so they
        # get a model fitted on EVERY labelled row — not a cross-validation
        # fold, which exists to estimate accuracy rather than to deliver an
        # answer. They were previously reported with a null prediction, which
        # was the one thing they must never be.
        final, pred_out = None, {}
        if to_predict:
            final = classify.fit(
                rows, labels,
                n_trees=int(rc.get("trees") or classify.DEFAULT_TREES),
                depth=int(rc.get("depth") or classify.DEFAULT_DEPTH),
                class_weight=(rc.get("class_weight") or None))
            thr = float(rc.get("threshold") or 0.5)
            for i in to_predict:
                xs = [float(frame[i].get(f)) for f in feats]
                pred_out[i] = (classify.predict(final, xs, positive, thr),
                               classify.predict_proba(final, xs))


        out = []
        for n, i in enumerate(keep):
            r = frame[i]
            row = {k: r.get(k) for k in series_keys} or {cols[0]: r.get(cols[0])}
            row[target] = labels[n]
            row["yhat"] = res["pred"][n]
            p = res["proba"][n] or {}
            row["proba"] = round(p.get(positive, max(p.values(), default=0.0)), 4)
            row["is_prediction"] = "no"
            out.append(row)
        for i in to_predict:
            r = frame[i]
            row = {k: r.get(k) for k in series_keys} or {cols[0]: r.get(cols[0])}
            yh, p = pred_out[i]
            row[target] = None
            row["yhat"] = yh
            row["proba"] = round(p.get(positive, max(p.values(), default=0.0)), 4)
            row["is_prediction"] = "yes"
            out.append(row)
        for i in incomplete:
            r = frame[i]
            row = {k: r.get(k) for k in series_keys} or {cols[0]: r.get(cols[0])}
            row[target] = label_of(r[target]) if r.get(target) is not None else None
            row["yhat"] = None
            row["proba"] = None
            row["is_prediction"] = "unscorable"
            out.append(row)
        return out, acc

    # ---------------------------------------------------- local recommender

    def _run_recommend(self, frame, series_keys, target, rc):
        """Item-item collaborative filtering over a user x item frame.

        Which key is the user and which is the item is a modelling decision, so
        it is read from the run config; falling back to key order is a
        convention, not a guess dressed up as one.
        """
        user_col = str(rc.get("user_key") or series_keys[0])
        item_col = str(rc.get("item_key") or
                       next(k for k in series_keys if k != user_col))
        k = int(rc.get("top_k") or recommend.DEFAULT_K)
        holdout = float(rc.get("holdout") or 0.3)
        folds = int(rc.get("folds") or 1)

        data = recommend.interactions(frame, user_col, item_col, target)
        if not data:
            raise ValueError(
                f"no interactions: '{user_col}' x '{item_col}' produced nothing "
                f"a recommender can read")

        acc = recommend.evaluate(data, k=k, holdout=holdout, folds=folds)
        recs, _, _ = recommend.fit_and_recommend(data, k=k)

        out = []
        for u, items in recs.items():
            for rank, (item, score) in enumerate(items, start=1):
                out.append({user_col: u, item_col: item, "rank": rank,
                            "score": score, "is_recommendation": "yes"})
        if acc:
            acc["user_key"] = user_col
            acc["item_key"] = item_col
            acc["users"] = len(data)
            acc["interactions"] = sum(len(v) for v in data.values())
        return out, acc

    # ------------------------------------------------------------- reading

    @staticmethod
    def _axes(meta, cols):
        """Which column is the clock, and which identify a series.

        Read from the frame's own metadata when the mapping layer supplied it;
        the frame is domain-blind, so guessing from column names here would
        reintroduce exactly the coupling the SPEC removed.
        """
        keys = [k for k in (meta.get("keys") or []) if k in cols]
        clock = None
        for k in keys:
            if ets.detect_grain(str((meta.get("t_min") or ""))) or k == "t":
                pass
        clock = meta.get("t_key") or ("t" if "t" in cols else None)
        if clock is None:
            for k in keys:
                sample = k
                if sample == "t":
                    clock = k
                    break
        series_keys = [k for k in keys if k != clock]
        return clock, series_keys

    @staticmethod
    def _target(rc, meta, cols, clock, series_keys):
        t = str(rc.get("target") or "").strip()
        if t:
            if t not in cols:
                raise ValueError(
                    f"run_config.target '{t}' is not a column of the frame "
                    f"({', '.join(cols)})")
            return t
        targets = [c for c in (rc.get("targets") or []) if c in cols]
        if targets:
            return targets[0]
        # nothing nominated: fall back to the first measured column, which is
        # what the frame has left once keys and the clock are removed
        spare = [c for c in cols if c != clock and c not in series_keys]
        if not spare:
            raise ValueError("the frame has no measured column to predict")
        return spare[0]

    @staticmethod
    def _series_count(frame, series_keys):
        if not series_keys:
            return 1
        return len({tuple(r.get(k) for k in series_keys) for r in frame})

    # ------------------------------------------------------- compatibility

    # Kinds with no time axis. Running the clock checks on one produces true
    # statements about the wrong question — "no key is a clock, so a horizon
    # means nothing" is not a warning to someone who never asked for a horizon.
    CLOCKLESS = ("classify", "cluster", "recommend", "rank", "regress")

    def _compat(self, engine, clock, series_keys, horizon, season, grain,
                frame, target, kind=""):
        """The checks the browser was doing, with the data now in hand."""
        checks, warnings = [], []
        clocked = kind not in self.CLOCKLESS

        def ok(label, good, detail=""):
            checks.append({"check": label, "ok": bool(good), "detail": detail})
            if not good:
                warnings.append(f"{label}: {detail}" if detail else label)

        if clocked:
            ok("frame has a time axis", clock is not None,
               "no key is a clock, so a horizon means nothing" if not clock else
               f"clock is '{clock}' at {grain} grain")
            if horizon:
                ok("horizon needs a clock", clock is not None,
                   f"horizon {horizon} was asked for but the frame has no clock")
        if clocked and season and grain:
            natural = NATURAL_SEASON.get(grain)
            ok("season length suits the grain",
               natural is None or season in (natural, 1) or season % natural == 0,
               f"season {season} at {grain} grain — the natural cycle is "
               f"{natural}" if natural else "")
        if engine in ("nixtla", "autogluon"):
            ok(f"{engine} accepts {len(series_keys)} entity key(s)",
               len(series_keys) <= 1,
               f"{engine} keys on a single id; this frame has "
               f"{len(series_keys)} — collapse them or pivot first")

        # the data-side checks the browser could never do
        n = self._series_count(frame, series_keys)
        per = len(frame) / n if n else 0
        if clocked and season:
            ok("enough history to fit the season", per >= 2 * season,
               f"{per:.0f} points per series but a season of {season} needs "
               f"{2 * season}")
        vals = [r.get(target) for r in frame]
        bad = next((v for v in vals if v is not None and not _isnum(v)), None)
        ok("target is numeric", bad is None,
           f"'{target}' holds text (e.g. {bad!r})" if bad is not None else "")
        if clocked:
            holes = self._gaps(frame, clock, series_keys)
            ok("the clock has no holes", not holes,
               f"{len(holes)} series skip periods — a model that advances one "
               f"period per row would be fitting a series that is not there"
               if holes else "")
        return warnings, checks

    @staticmethod
    def _gaps(frame, clock, series_keys):
        if not clock:
            return []
        groups = {}
        for r in frame:
            groups.setdefault(tuple(r.get(k) for k in series_keys), []).append(r)
        out = []
        for combo, g in groups.items():
            idx = sorted(filter(None, (ets.period_index(str(r.get(clock)))
                                       for r in g)))
            if len(idx) > 1 and max(b - a for a, b in zip(idx, idx[1:])) > 1:
                out.append(combo)
        return out

    # ------------------------------------------------------------ dialect

    def _to_dialect(self, frame, engine, clock, series_keys, target, features,
                    meta):
        """Rename the frame into the engine's vocabulary. Values are untouched."""
        d = DIALECT[engine]
        rows = []
        if engine == "recbole" and len(series_keys) >= 2:
            for r in frame:
                rows.append({d["series"]: r.get(series_keys[0]),
                             d["item"]: r.get(series_keys[1]),
                             d["target"]: r.get(target)})
            return rows
        for r in frame:
            row = {}
            if series_keys:
                row[d["series"]] = "|".join(str(r.get(k)) for k in series_keys)
            elif engine != "local":
                row[d["series"]] = "all"        # both libraries require an id
            if clock:
                row[d["clock"]] = r.get(clock)
            row[d["target"]] = r.get(target)
            for f in features:
                row[f] = r.get(f)
            rows.append(row)
        return rows

    # -------------------------------------------------------- local engine

    def _run_local(self, frame, clock, series_keys, target, horizon, season):
        """Fit stdlib exponential smoothing per series and forecast forward."""
        if not clock:
            raise ValueError(
                "engine 'local' forecasts along a clock, and this frame has no "
                "time key — use mode 'prepare', or compile a spec with a bin: key")
        h = horizon or 4
        groups = {}
        for r in frame:
            groups.setdefault(tuple(r.get(k) for k in series_keys), []).append(r)

        out, per_series = [], []
        for combo, g in groups.items():
            g = sorted(g, key=lambda r: str(r.get(clock)))
            label = "|".join(str(c) for c in combo) or "all"
            y = [float(r[target]) for r in g
                 if r.get(target) is not None and _isnum(r[target])]
            if len(y) < 4:
                out.extend({**r, "yhat": None, "is_forecast": "no"} for r in g)
                per_series.append({"series": label, "n": len(y),
                                   "note": "too short to fit (needs 4+ points)"})
                continue

            ev = (ets.evaluate_rolling(y, season, horizon=h)
                  or ets.evaluate(y, season, holdout=h))
            state = ets.fit(y, season)
            for r, f in zip(g, state["fitted"]):
                out.append({**r, "yhat": round(f, 4), "is_forecast": "no"})
            labels = ets.next_periods(g[-1].get(clock), h)
            for lab, v in zip(labels, ets.forecast(state, h)):
                out.append({**{k: c for k, c in zip(series_keys, combo)},
                            clock: lab, target: None,
                            "yhat": round(v, 4), "is_forecast": "yes"})

            entry = {"series": label, "n": len(y),
                     "model": ("holt-winters" if state["m"] else
                               "holt" if state["beta"] is not None else "ses"),
                     "alpha": state["alpha"], "beta": state["beta"],
                     "gamma": state["gamma"]}
            if ev:
                entry.update({k: v for k, v in ev.items() if k in (
                    "mape", "mae", "rmse", "mase", "naive_mape", "naive_mae",
                    "skill_vs_naive", "holdout", "folds", "points_scored",
                    "method", "mape_usable")})
            per_series.append(entry)

        scored = [s for s in per_series if s.get("mae") is not None]
        acc = {"per_series": per_series, "scored_series": len(scored),
               "horizon": h, "season_length": season}
        mape_ok = all(s.get("mape_usable", True) for s in scored) if scored else None
        keys = ["mae", "naive_mae", "rmse", "mase", "skill_vs_naive"]
        if mape_ok:
            keys = ["mape", "naive_mape"] + keys
        for k in keys:
            v = _avg(scored, k)
            if v is not None:
                acc[k] = round(v, 3)
        acc["mape_usable"] = mape_ok
        if mape_ok is False:
            acc["mape_suppressed"] = (
                "a series reaches zero, so a percentage error is undefined "
                "there — MAE and MASE carry the judgement instead")
        return out, acc

    @staticmethod
    def _verdict(acc):
        """trusted / weak / unpredictable, from whichever metric survived."""
        if not acc or not acc.get("scored_series"):
            return "not scored — every series was too short for a holdout"
        skill, mape, mase = (acc.get("skill_vs_naive"), acc.get("mape"),
                             acc.get("mase"))
        if skill is None:
            return "scored, but the baseline was exact — nothing to compare"
        if mape is not None and mape < 1:
            return "exact — the series barely moves; model and baseline agree"
        accurate = (mape < 15) if mape is not None else (
            mase < 1.0 if mase is not None else None)
        if skill > 20 and accurate:
            return "trusted — beats naive by a clear margin"
        if skill > 20 and accurate is False:
            return "mixed — clearly beats the baseline, but the error is large"
        if skill > 0:
            return "weak — beats naive, but not by much"
        return "unpredictable — does not beat carrying the last value forward"

    # -------------------------------------------------------------- schema

    def schema(self):
        SSP = BaseService.ServiceSchemaProperty
        return [
            SSP(key="mode", type="enum", default="run",
                enum=[ParameterEnum(label=m, value=m) for m in MODES],
                description="run fits the local model; prepare only emits the dialect"),
            SSP(key="run_config", type="object", required=True,
                description="engine, target, features, horizon, season_length"),
            SSP(key="frame_meta", type="object",
                description="the mapping layer's metadata — keys, grain, t_key"),
        ]

    # ----------------------------------------------------------- self test

    def self_test(self):
        checks = {}

        # a two-series monthly frame with a clean annual shape
        frame = []
        for s, base in (("a", 100), ("b", 500)):
            for i in range(36):
                y, m = 2020 + i // 12, i % 12 + 1
                frame.append({"series_id": s, "t": f"{y}-{m:02d}",
                              "y": base + i * 2 + (10 if m in (6, 7) else 0)})
        meta = {"keys": ["series_id", "t"], "t_key": "t", "grain": "month"}

        out, rep = self._adapt(frame, {
            "run_config": {"engine": "local", "target": "y", "horizon": 6,
                           "season_length": 12},
            "frame_meta": meta})
        checks["local engine returns predictions"] = any(
            r["is_forecast"] == "yes" for r in out)
        checks["one forecast block per series"] = (
            len([r for r in out if r["is_forecast"] == "yes"]) == 12)
        checks["future periods continue the clock"] = (
            {r["t"] for r in out if r["is_forecast"] == "yes"} ==
            {f"2023-{m:02d}" for m in range(1, 7)})
        checks["accuracy is measured, not asserted"] = (
            rep["accuracy"]["scored_series"] == 2
            and rep["accuracy"].get("skill_vs_naive") is not None)
        checks["a verdict is produced"] = bool(rep["verdict"])
        checks["series counted from the keys"] = rep["series_count"] == 2

        # dialects rename and nothing else
        nix, rn = self._adapt(frame, {
            "run_config": {"engine": "nixtla", "target": "y", "horizon": 6},
            "frame_meta": meta})
        checks["nixtla dialect renames the three axes"] = (
            set(nix[0]) == {"unique_id", "ds", "y"})
        checks["nixtla is prepared, not run"] = rn["mode"] == "prepare"
        checks["dialect preserves the values"] = (
            nix[0]["y"] == frame[0]["y"] and nix[0]["ds"] == frame[0]["t"])
        ag, _ = self._adapt(frame, {
            "run_config": {"engine": "autogluon", "target": "y"},
            "frame_meta": meta})
        checks["autogluon dialect differs"] = (
            set(ag[0]) == {"item_id", "timestamp", "target"})

        # the run config, not the frame, chooses what is predicted
        two = [{**r, "z": r["y"] * 3} for r in frame]
        a, _ = self._adapt(two, {"run_config": {"engine": "local", "target": "y",
                                                "horizon": 3, "season_length": 12},
                                 "frame_meta": meta})
        b, _ = self._adapt(two, {"run_config": {"engine": "local", "target": "z",
                                                "horizon": 3, "season_length": 12},
                                 "frame_meta": meta})
        checks["swapping the target changes the prediction"] = (
            [r["yhat"] for r in a if r["is_forecast"] == "yes"] !=
            [r["yhat"] for r in b if r["is_forecast"] == "yes"])
        checks["swapping the target does not touch the frame"] = (
            [{k: v for k, v in r.items() if k in ("series_id", "t", "y", "z")}
             for r in a if r["is_forecast"] == "no"] ==
            [{k: v for k, v in r.items() if k in ("series_id", "t", "y", "z")}
             for r in b if r["is_forecast"] == "no"])

        # compatibility reads the data, not only the documents
        _, rc = self._adapt(frame, {
            "run_config": {"engine": "local", "target": "y", "horizon": 6,
                           "season_length": 7},
            "frame_meta": meta})
        checks["season vs grain is checked"] = any(
            not c["ok"] and "season" in c["check"] for c in rc["checks"])
        gappy = [r for r in frame if r["t"] not in ("2020-05", "2021-08")]
        _, rg = self._adapt(gappy, {
            "run_config": {"engine": "local", "target": "y", "horizon": 3,
                           "season_length": 12},
            "frame_meta": meta})
        checks["clock holes are reported"] = any(
            not c["ok"] and "holes" in c["check"] for c in rg["checks"])
        _, rk = self._adapt(frame, {
            "run_config": {"engine": "nixtla", "target": "y"},
            "frame_meta": {**meta, "keys": ["series_id", "region", "t"]}})
        checks["entity-key count is checked per engine"] = any(
            "entity key" in c["check"] for c in rk["checks"])

        # ---- recommendation: a grid of two keys and no clock ----
        grid = []
        for u in range(60):
            taste = u % 3
            for it in range(taste * 6, taste * 6 + 4):
                grid.append({"k_user": f"u{u}", "k_item": f"i{it}", "w": 1.0})
        gmeta = {"keys": ["k_user", "k_item"], "t_key": None, "grain": None}

        rec, rr = self._adapt(grid, {
            "run_config": {"engine": "local", "target": "w", "top_k": 3},
            "frame_meta": gmeta})
        checks["a keyless-clock grid routes to the recommender"] = (
            rr["task"] == "recommend")
        checks["recommendations come back ranked"] = (
            bool(rec) and rec[0]["rank"] == 1
            and all(r["is_recommendation"] == "yes" for r in rec))
        checks["top_k is honoured"] = all(
            r["rank"] <= 3 for r in rec)
        checks["structure is found: it beats popularity"] = (
            rr["accuracy"]["lift_over_popularity"] > 20)
        checks["a recommendation verdict is produced"] = "popular" in rr["verdict"]
        checks["the baseline is reported alongside"] = (
            "popular_precision_at_k" in rr["accuracy"])

        # Noise must not SYSTEMATICALLY look like skill. Averaged over seeds,
        # because one draw of random data swings between -36% and +86% lift -
        # a single-seed assertion here failed on variance and told us nothing.
        # This is the check that caught a genuinely biased split: holding back
        # each user's alphabetically-last items handicapped the popularity
        # baseline, and noise scored a steady ~38%.
        import random as _r
        lifts = []
        for seed in range(12):
            _r.seed(seed)
            noise = [{"k_user": f"u{u}", "k_item": f"i{_r.randint(0, 17)}",
                      "w": 1.0} for u in range(60) for _ in range(4)]
            _, rn = self._adapt(noise, {
                "run_config": {"engine": "local", "target": "w", "top_k": 3},
                "frame_meta": gmeta})
            lift = (rn.get("accuracy") or {}).get("lift_over_popularity")
            if lift is not None:
                lifts.append(lift)
        mean_lift = sum(lifts) / len(lifts) if lifts else 0.0
        checks["noise does not systematically beat popularity"] = abs(mean_lift) < 25

        # a forecast frame must still route to forecasting
        checks["a clocked frame still forecasts"] = rep["task"] == "forecast"

        # recbole speaks its own dialect
        rb, _ = self._adapt(grid, {
            "run_config": {"engine": "recbole", "target": "w"},
            "frame_meta": gmeta})
        checks["recbole dialect names both keys and types"] = (
            set(rb[0]) == {"user_id:token", "item_id:token", "rating:float"})

        checks["a forecast declares forecasting knobs"] = (
            "horizon" in rep["knobs"] and "top_k" not in rep["knobs"])
        checks["a recommendation declares its own"] = (
            "top_k" in rr["knobs"] and "horizon" not in rr["knobs"])
        _, ri = self._adapt(grid, {
            "run_config": {"engine": "local", "target": "w", "top_k": 3,
                           "horizon": 12, "season_length": 4},
            "frame_meta": gmeta})
        checks["knobs the task cannot read are reported, not silently eaten"] = (
            set(ri["ignored_knobs"]) >= {"horizon", "season_length"})

        # ---- classification: a labelled table, no clock ----
        import random as _rc
        cls, cmeta = [], {"keys": ["k_row"], "t_key": None, "grain": None,
                          "classes": {"y": {"0": "N", "1": "Y"}}}
        _rc.seed(5)
        for i in range(120):
            risk = i % 4 == 0
            cls.append({"k_row": f"r{i}",
                        "y": 1.0 if risk else 0.0,
                        "f1": _rc.uniform(35, 68) if risk else _rc.uniform(70, 99),
                        "f2": _rc.uniform(3, 14) if risk else _rc.uniform(0, 2)})
        crc = {"engine": "local", "kind": "classify", "target": "y",
               "features": ["f1", "f2"], "folds": 5, "positive_class": "Y"}
        out, rr = self._adapt(cls, {"run_config": crc, "frame_meta": cmeta})

        checks["kind routes to the classifier"] = rr["task"] == "classify"
        checks["separable classes are learned"] = (
            rr["accuracy"]["lift_over_baseline"] > 10)
        checks["the majority-class baseline is reported"] = (
            rr["accuracy"]["majority_baseline"] > 0
            and rr["accuracy"]["majority_class"] == "N")
        checks["per-class precision and recall are reported"] = (
            set(rr["accuracy"]["per_class"]) == {"N", "Y"}
            and "recall" in rr["accuracy"]["per_class"]["Y"])
        checks["a confusion matrix is produced"] = (
            sum(sum(v.values()) for v in rr["accuracy"]["confusion"].values()) == 120)
        checks["class ids come back as names"] = (
            out[0]["y"] in ("N", "Y") and out[0]["yhat"] in ("N", "Y"))
        checks["classify declares its own knobs"] = (
            "threshold" in rr["knobs"] and "horizon" not in rr["knobs"])

        # a lower threshold must trade precision for recall, not improve both
        lo, _ = self._adapt(cls, {"frame_meta": cmeta, "run_config": {
            **crc, "threshold": 0.2, "class_weight": "balanced"}})
        hi, _ = self._adapt(cls, {"frame_meta": cmeta, "run_config": {
            **crc, "threshold": 0.8, "class_weight": "balanced"}})
        n_lo = sum(1 for r in lo if r["yhat"] == "Y")
        n_hi = sum(1 for r in hi if r["yhat"] == "Y")
        checks["a lower threshold flags more rows"] = n_lo >= n_hi

        # noise must not look like skill, averaged over seeds
        lifts = []
        for seed in range(6):
            _rc.seed(200 + seed)
            noise = [{"k_row": f"r{i}", "y": float(_rc.randint(0, 1)),
                      "f1": _rc.random(), "f2": _rc.random()} for i in range(120)]
            _, rn = self._adapt(noise, {"frame_meta": cmeta, "run_config": crc})
            lifts.append(rn["accuracy"]["lift_over_baseline"] or 0.0)
        checks["noise does not systematically beat the baseline"] = (
            sum(lifts) / len(lifts) < 10)

        # failures are loud
        def cfails(frame, rc_over, needle):
            try:
                self._adapt(frame, {"frame_meta": cmeta,
                                    "run_config": {**crc, **rc_over}})
                return False
            except ValueError as e:
                return needle in str(e)
        # positive_class is validated first, so drop it here — otherwise this
        # asserts the wrong refusal and passes for the wrong reason.
        checks["a single-class column is rejected"] = cfails(
            [{**r, "y": 0.0} for r in cls], {"positive_class": ""},
            "nothing to separate")
        checks["too few rows is rejected"] = cfails(
            cls[:3], {}, "too few")
        checks["an unknown positive_class is rejected"] = cfails(
            cls, {"positive_class": "Z"}, "is not one of the labels")

        # failures are loud
        def fails(params, needle):
            try:
                self._adapt(frame, params)
                return False
            except ValueError as e:
                return needle in str(e)
        checks["unknown engine is rejected"] = fails(
            {"run_config": {"engine": "prophet"}}, "engine must be one of")
        checks["a target not in the frame is rejected"] = fails(
            {"run_config": {"engine": "local", "target": "nope"},
             "frame_meta": meta}, "is not a column of the frame")
        checks["an empty frame is rejected"] = (
            self._adapt([], {"run_config": {}}) if False else True)
        try:
            self._adapt([], {"run_config": {"engine": "local"}})
            checks["an empty frame is rejected"] = False
        except ValueError as e:
            checks["an empty frame is rejected"] = "no frame" in str(e)

        return {"checks": checks}


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _avg(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None
