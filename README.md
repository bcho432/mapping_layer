# mapping layer — pipeline lab

A local bench for the Decision Engine's three services. Type an ask, pick some
data, watch a canonical frame come out the other end.

```
prompt + CSV  ->  profile-generator (suggest -> finalize)
              ->  spec-compiler      (obligation walk -> SPEC)
              ->  mapping-layer      (as-of group-by  -> canonical frame)
```

Nothing is sent to an engine. There is no engine.

## Run it

```bash
python server.py        # http://127.0.0.1:8770
python server.py 9001   # another port
```

Stdlib only — no install, no venv, no requirements file. The real `spl.core`
needs `pydantic-settings`, so the four classes the services actually touch are
stubbed; `spl.services` is mapped at `./services`, which keeps each service's own
`function_library` / `derive_library` copy separate exactly as in production.

The badge in the top right runs all three self-tests on load (123 checks). If it
is red, the services are broken, not the lab.

## Layout

```
server.py            stdlib HTTP server + the spl.core stub + the pipeline
app.html             the whole UI, no build step
datasets/*.csv       drop a CSV here and it appears in the picker
services/            profile_generator, spec_compiler, mapping_layer
```

### services/ is a copy

These packages are copied from the SPL repo, which is where they ship:

```
C:\Users\Owner.DESKTOP-13CAF6J\WebstormProjects\ss_spl\src\spl\services\
```

Two trees means they can drift. Decide per change which is the source:

```bash
SS=~/WebstormProjects/ss_spl/src/spl/services

# see what differs
for s in profile_generator spec_compiler mapping_layer; do diff -r "$SS/$s" "services/$s"; done

# refresh the bench from the repo (repo is source of truth)
for s in profile_generator spec_compiler mapping_layer; do cp -r "$SS/$s/." "services/$s/"; done

# push a bench edit back to the repo (bench is source of truth)
for s in profile_generator spec_compiler mapping_layer; do cp -r "services/$s/." "$SS/$s/"; done
```

Anything meant to ship has to land in `ss_spl` — the publisher only ever sees
that tree. On `ss_spl`, `main` is protected: branch, then merge request.

## What each panel is for

**the ask** — a plain-language goal, a dataset, and the frame's column naming.
`v1` keeps the legacy `series_id / t / y / x_*`; `logical` names columns after
the profile. Same numbers either way — that is the point of the toggle.

**run config** — lives entirely in the browser and is echoed back untouched.
Change the horizon, the season, the engine, **or the target column**, and re-run:
**every cell of the frame must be identical.** That is the invariant the
SPEC/run-config split exists to protect, and this panel is how you check it
rather than assume it.

The target picker is the point. The SPEC aggregates; it does not say which
column is the thing being predicted, because that is a per-run decision. The
compiler still reports what the profile *intended* as
`metadata.run_config_hint`, which is what prefills the picker — a suggestion you
can override without recompiling anything.

**adapter compatibility** — the checks that will move into the engine adapter
once the run config is real: season length against the grain, horizon against
whether the SPEC has a time axis at all, engine against the number of entity
keys. Updates live as you edit the run config.

**the canonical frame** — the actual deliverable. Key columns are tinted; nulls
are shown as nulls rather than blanks, because a null target with a live feature
is meaningful (see below). Stats underneath cover rows in/out, drop reasons,
series count, join gaps and the leakage counter.

**edit & re-run** — the profile, the binding and the SPEC are all editable.
Re-running from the SPEC skips both upstream services, which is the fastest way
to poke directly at frame generation.

## Things worth trying

- **Clustering.** Edit the SPEC to one key and no `time_from`, then set the
  target picker to *(none — unsupervised)*. It works, and note that the SPEC
  itself is unchanged from the supervised case: nothing in it ever said which
  column was a target. Under the old pipeline this died at
  `spec_compiler.py:246`, and even with that removed the mapping layer returned
  an empty frame because the `got_any` gate only counted targets.

- **A composite key.** Add a second entry to `keys[]`. The old `series_from` was
  a single string, so this was unrepresentable.

- **The leakage guard.** Add `"validity": {"arrival_from": "reported_dt"}` to the
  awards SPEC and set an aggregate's `"anchor": "arrival"`. The Sept 27 award was
  not reported until Oct 30, so that column's value is re-filed forward into Q4 —
  producing a Q4 row with a **null in the event-anchored column and a live value
  in the arrival-anchored one**. The counter reads
  `features_refiled_forward: 1`. That row existing at all is the guard and the
  `got_any` fix working together.

- **Two anchors, one measure.** Aggregate the same column twice, once with
  `anchor: "event"` and once with `anchor: "arrival"`. You get both the as-it-
  happened and the as-you-knew-it series side by side, from one pass. This is
  what `anchor` buys over the old `role` + `availability` pair, which could only
  express it by pretending one of them was a target.

- **Role is gone.** Put `"role": "target"` on an aggregate. It is rejected, and
  the message points at the run config.

- **A gap, not a default.** Set a key's `via` to `bin:fortnight`. It fails loudly
  and lists the catalog. The old code silently defaulted a missing grain to
  weekly.

## Datasets

`datasets/awards.csv` — federal awards, three categories, nine months, with a
`reported_dt` arrival column for the leakage guard. HHI is verifiable by hand:
aerospace Q1 is 60/30/10 → 4600; it_services Q1 is four vendors at 25% → 2500;
pharma Q1 nets to Helios 60M / Ironwood 40M → 5200.

`datasets/retail.csv` — two stores, two families, eight weeks.

Drop any `.csv` into `datasets/` and it appears in the picker. Or choose
*paste your own*.

## A caveat about the first stage

The profile-generator is heuristics over column names, and it is the weakest
link in the chain by design — it exists to be corrected by a human before
signing. It reads any column whose sample parses as a number as a candidate
measure, so a numeric-looking code column can be picked as the thing to
aggregate. The sample data avoids that, but your own data may not; if the
emitted SPEC looks wrong, check the profile panel first, fix it there, and
re-run from the profile. A wrong frame from a wrong profile is the pipeline
working correctly.

## Not built yet

The run config is frontend-only on purpose. The next piece is the engine
adapter: per-engine dialect JSON (`unique_id`/`ds` for Nixtla,
`item_id`/`timestamp` + `known_covariates` for AutoGluon), the future-row
scaffold built from the horizon, and the compatibility checks moved server-side
where they can also inspect the data rather than only the documents.
