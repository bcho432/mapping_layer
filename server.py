"""pipeline-lab: a local bench for the Decision Engine's three services.

    prompt + data  ->  profile-generator (suggest -> finalize)
                   ->  spec-compiler      (obligation walk -> SPEC)
                   ->  mapping-layer      (as-of group-by -> canonical frame)

Nothing is sent to an engine. The run config lives in the browser and is echoed
back untouched, which is the point: the frame is a pure function of the SPEC, so
changing horizon or season must not change a single cell.

Stdlib only. The real spl.core needs pydantic-settings, so the four modules the
services actually touch are stubbed; spl.services is mapped at ./services, which
keeps each service's own function_library / derive_library copy separate exactly
as it is in production.

The packages under ./services are a COPY of the ones in the ss_spl repo at
src/spl/services. This project is the bench; that repo is where they ship. See
README.md for the sync command and which direction to run it.

    python server.py [port]
"""

import csv
import io
import json
import os
import sys
import time
import traceback
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICES = os.path.join(HERE, "services")
DATASETS = os.path.join(HERE, "datasets")


# ------------------------------------------------------------------- stubs

class _Logger:
    def __init__(self):
        self.lines = []

    def _add(self, level, msg):
        self.lines.append(f"{level}: {msg}")

    def info(self, msg, *a, **k):
        self._add("info", msg)

    def warning(self, msg, *a, **k):
        self._add("warning", msg)

    def error(self, msg, *a, **k):
        self._add("error", msg)


class ServiceSchemaProperty(dict):
    def __init__(self, **kw):
        super().__init__(**kw)


class ParameterEnum(dict):
    def __init__(self, **kw):
        super().__init__(**kw)


class RunRequest:
    def __init__(self, data=None, parameters=None):
        self.data = data
        self.parameters = parameters


class RunResponse:
    def __init__(self, data=None, metadata=None):
        self.data = data
        self.metadata = metadata


class BaseService:
    ServiceSchemaProperty = ServiceSchemaProperty
    RunRequest = RunRequest
    RunResponse = RunResponse
    logger = _Logger()


def _install():
    """Stub spl.core; point spl.services at the real package directory."""
    def mod(name, path=None):
        m = types.ModuleType(name)
        if path is not None:
            m.__path__ = path
        sys.modules[name] = m
        return m

    spl = mod("spl", [])
    mod("spl.core", [])
    mod("spl.core.base_service", [])
    sys.modules["spl.core.base_service.base_service_class"] = \
        mod("spl.core.base_service.base_service_class")
    sys.modules["spl.core.base_service.base_service_class"].BaseService = BaseService
    sys.modules["spl.core.service_types"] = mod("spl.core.service_types")
    sys.modules["spl.core.service_types"].ParameterEnum = ParameterEnum

    services = mod("spl.services", [SERVICES])
    spl.services = services


_install()
import importlib  # noqa: E402  (must follow the stub install)

PG = importlib.import_module("spl.services.profile_generator.profile_generator")
SC = importlib.import_module("spl.services.spec_compiler.spec_compiler")
ML = importlib.import_module("spl.services.mapping_layer.mapping_layer")
EA = importlib.import_module("spl.services.engine_adapter.engine_adapter")
FL = importlib.import_module("spl.services.mapping_layer.function_library")
# The engines the model may choose from come from the same table the compiler
# validates against, so the prompt cannot describe a pipeline that no longer
# exists — or miss one that now does.
TL = importlib.import_module("spl.services.spec_compiler.task_library")
import llm_profile  # bench-side only: the services stay stdlib-only

profile_generator = PG.ProfileGeneratorService()
spec_compiler = SC.SpecCompilerService()
mapping_layer = ML.MappingLayerService()
engine_adapter = EA.EngineAdapterService()


# -------------------------------------------------------------------- data

def read_csv(text):
    """CSV text -> (rows, schema) with the sample + cardinality the
    profile-generator's heuristics read."""
    rdr = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    rows = [dict(r) for r in rdr]
    schema = []
    for c in (rdr.fieldnames or []):
        vals = [r.get(c) for r in rows if str(r.get(c) or "").strip()]
        schema.append({
            "name": c,
            "sample": vals[0] if vals else None,
            "cardinality": len(set(vals)),
        })
    return rows, schema


def datasets():
    out = []
    if os.path.isdir(DATASETS):
        for fn in sorted(os.listdir(DATASETS)):
            if fn.endswith(".csv"):
                with open(os.path.join(DATASETS, fn), encoding="utf-8") as f:
                    out.append({"id": fn[:-4], "csv": f.read()})
    return out


# ---------------------------------------------------------------- pipeline

class Halt(Exception):
    def __init__(self, stages):
        self.stages = stages


def run(payload):
    stages = []

    def step(sid, title, fn):
        t0 = time.perf_counter()
        try:
            value, detail = fn()
        except Exception as e:                      # noqa: BLE001 - surfaced to the UI
            stages.append({
                "id": sid, "title": title, "ok": False,
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc().splitlines()[-6:],
            })
            raise Halt(stages)
        stages.append({
            "id": sid, "title": title, "ok": True,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "output": value, "detail": detail,
        })
        return value

    csv_text = payload.get("csv") or ""
    if not csv_text.strip():
        raise ValueError("no data: pick a dataset or paste CSV")
    rows, schema = read_csv(csv_text)
    if not rows:
        raise ValueError("the CSV parsed to zero rows")

    dataset = str(payload.get("dataset") or "dataset")
    naming = str(payload.get("naming") or "v1")

    # 1 + 2 -- profile-generator, unless the caller edited the profile by hand
    profile = payload.get("profile")
    binding = payload.get("binding")
    if not isinstance(profile, dict):
        draft = step(
            "suggest", "profile-generator (suggest)",
            lambda: profile_generator._suggest({
                "schema": schema, "goal": payload.get("goal") or "",
                "dataset": dataset, "grain": payload.get("grain") or "",
            }))
        binding = binding if isinstance(binding, dict) else \
            stages[-1]["detail"]["binding_stub"]

        # The model sits BETWEEN the two deterministic halves. It only ever
        # touches a draft, and whatever it returns still has to survive
        # finalize — which is what makes a bad reply a fallback, not a bug.
        # Which reader parses the goal is a pipeline choice, not a modelling
        # one, so it rides alongside `naming` rather than in the run config —
        # it changes how the SPEC is authored, not what the engine does with
        # the frame. Absent means "use the model if a credential exists".
        goal = payload.get("goal") or ""
        use_llm = payload.get("llm")
        use_llm = True if use_llm is None else bool(use_llm)
        if goal.strip() and use_llm:
            def enrich():
                prof, rep = llm_profile.enrich(
                    draft, schema, goal, FL.catalog(), TL.catalog(),
                    binding_stub=binding)
                return prof, rep
            enriched = step("llm", "llm (read the goal)", enrich)
            llm_report = stages[-1]["detail"]
            if llm_report.get("used"):
                draft = enriched
                binding = llm_profile.rebind(binding, draft, dataset)

        elif goal.strip():
            stages.append({
                "id": "llm", "title": "llm (read the goal)", "ok": True, "ms": 0.0,
                "output": None,
                "detail": {"used": False, "model": llm_profile.MODEL,
                           "changes": [], "off": True,
                           "credential": llm_profile.credential_source() or "none",
                           "reason": "turned off — the keyword scaffold is "
                                     "reading the goal"},
            })

        profile = step(
            "finalize", "profile-generator (finalize)",
            # The schema rides along so finalize can check that every column a
            # derive reads as a date actually holds one. It is the last gate a
            # hand-edited profile passes, and the only one that has the values.
            lambda: profile_generator._finalize({"draft": draft, "schema": schema}))
    if not isinstance(binding, dict):
        raise ValueError("no binding: the suggest step normally provides the stub")

    # 3 -- spec-compiler, unless the caller edited the SPEC by hand
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        spec = step(
            "compile", "spec-compiler",
            lambda: spec_compiler._compile(profile, binding, naming))

    # 4 -- mapping-layer
    def execute():
        resolved = mapping_layer._resolve_spec(spec)
        frame, meta = mapping_layer._execute(rows, resolved)
        return frame, meta

    frame = step("frame", "mapping-layer", execute)
    meta = stages[-1]["detail"]

    # 5 -- engine-adapter. The frame is the deliverable; this is what finally
    # does something with it. Skipped when the caller asks for the frame only.
    rc = dict(payload.get("run_config") or {})
    crc = next((st.get("detail", {}).get("run_config") for st in stages
                if st["id"] == "compile" and isinstance(st.get("detail"), dict)),
               None) or {}
    # `kind` rides along with the target/feature split: the adapter routes on it,
    # and without it a classification frame falls through to the forecaster and
    # is refused for having no clock — which is a true statement about the wrong
    # question.
    for k in ("kind", "target", "targets", "features"):
        if not rc.get(k) and crc.get(k):
            rc[k] = crc[k]
    predictions, engine_meta = None, None
    if payload.get("engine") is not False and rc:
        def adapt():
            return engine_adapter._adapt(frame, {"run_config": rc,
                                                 "frame_meta": meta,
                                                 "mode": payload.get("mode")})
        try:
            predictions = step("engine", "engine-adapter", adapt)
            engine_meta = stages[-1]["detail"]
        except Halt:
            predictions, engine_meta = None, None

    cols = []
    for r in frame:
        for k in r:
            if k not in cols:
                cols.append(k)

    return {
        "ok": True,
        "stages": stages,
        "profile": profile,
        "binding": binding,
        "spec": spec,
        "schema": schema,
        "rows_in": len(rows),
        "frame": {"columns": cols, "rows": frame},
        "frame_meta": meta,
        # echoed straight back, never used: proof it cannot touch the frame
        "run_config": payload.get("run_config"),
        # what the compiler scaffolded: which emitted columns the profile
        # nominated as target(s). This used to live in the SPEC as
        # aggregates[].role; it is a modelling choice, so it belongs here.
        "predictions": predictions,
        "engine_meta": engine_meta,
        "llm": next((st.get("detail") for st in stages if st["id"] == "llm"), None),
        "compiled_run_config": next(
            (st.get("detail", {}).get("run_config") for st in stages
             if st["id"] == "compile" and isinstance(st.get("detail"), dict)),
            None),
    }


# ------------------------------------------------------------------ server

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "app.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/datasets":
            return self._send(200, json.dumps(datasets()))
        if path == "/api/selftest":
            out = {}
            for sid, svc in (("profile-generator", profile_generator),
                             ("spec-compiler", spec_compiler),
                             ("mapping-layer", mapping_layer),
                             ("engine-adapter", engine_adapter)):
                r = svc.self_test()
                checks = r.get("checks", {})
                out[sid] = {
                    "passed": sum(1 for v in checks.values() if v),
                    "total": len(checks),
                    "failed": [k for k, v in checks.items() if not v],
                }
            # The LLM stage is bench-side, so it is not a service — but it is
            # the one stage that can rewrite the whole profile, and it was the
            # only one with no tests at all. Its offline half runs here, with
            # the two stages below it wired in, so an accepted reply is proved
            # to survive them rather than assumed to.
            r = llm_profile.self_test(
                TL.catalog(), FL.catalog(),
                finalize=profile_generator._finalize,
                compile_fn=spec_compiler._compile)
            checks = r.get("checks", {})
            out["llm-profile"] = {
                "passed": sum(1 for v in checks.values() if v),
                "total": len(checks),
                "failed": [k for k, v in checks.items() if not v],
            }
            return self._send(200, json.dumps(out))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path.split("?")[0] != "/api/run":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except ValueError as e:
            return self._send(400, json.dumps({"ok": False, "error": f"bad JSON: {e}"}))
        try:
            return self._send(200, json.dumps(run(payload), default=str))
        except Halt as h:
            return self._send(200, json.dumps({"ok": False, "stages": h.stages},
                                              default=str))
        except Exception as e:                       # noqa: BLE001
            return self._send(200, json.dumps({
                "ok": False, "stages": [],
                "error": f"{type(e).__name__}: {e}"}, default=str))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"pipeline-lab  ->  http://127.0.0.1:{port}")
    print(f"  services from {SERVICES}")
    print("  ctrl-c to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
