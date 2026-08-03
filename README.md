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

The badge in the top right runs every service's self-tests on load (247 checks).
If it is red, the services are broken, not the lab.

## The profile is a core and a task

One question sorts every field of a profile: with no engine attached, does this
still describe something real about the data?

```jsonc
{ "metrics":    [{ "name": "rentals", "function": "sum", "measure": "cnt" }],
  "dimensions": [{ "name": "store" }],

  "task": { "kind":   "forecast",
            "clock":  { "event": "dteday", "grain": "day" },
            "keys":   ["store"],
            "target": "rentals",
            "covariates": [{ "metric": "promo", "availability": "known_ahead" }] } }
```

A metric is now purely **how to compute a quantity** — no role, no availability,
no time — which is true whether or not an engine ever runs. Everything that goes
meaningless without an engine is in the task.

That inversion is what freezes the metric schema. A role stamped on a metric
grows its enum every time an engine is added: a recommender wants
`role: "signal"`, a classifier wants `role: "label"`. A task holding a *pointer*
to a metric does not.

Which pointer each kind uses, whether it has a clock, and how many keys it takes
all live in one table — `task_library.py`, a CSV with an identical embedded
fallback, the same pattern as `function_library` and `derive_library`. Both the
profile-generator (which writes tasks) and the spec-compiler (which lowers them)
validate against it, so they cannot drift apart. It replaced a hand-kept
`CONTRACT` dict in one service and a `KNOWN_KINDS`/`CLOCKED_KINDS` pair in the
other — two tables describing the same seven questions, with nothing forcing
them to agree. **Adding an engine is a row.**

| kind | requires | permits | keys |
|---|---|---|---|
| `forecast` | clock, target | keys, covariates | 0..4 |
| `detect_anomaly` | signal | clock, keys, covariates | 0..4 |
| `classify` | label, keys | clock, covariates | 1..4 |
| `regress` | target, keys | clock, covariates | 1..4 |
| `cluster` | keys | clock, covariates | 1..4 |
| `recommend` | keys, signal | covariates | 2..2 |
| `rank` | keys, signal | covariates | 2..2 |

A field a kind has no concept of is an **error**, not a silently ignored hint —
otherwise every engine would have to know to skip it, and nothing would catch a
recommender that set a grain by mistake.

**The kind comes from the goal text**, by deterministic regex in a fixed order,
in front of the language model. Even a total LLM outage still produces a
correctly-shaped task. A kind that *requires* a clock always gets one; a kind
that merely *permits* one gets it only if the goal asks for time — otherwise
"segment customers" would quietly become "segment customers per month", which is
a different question.

**A flat profile still compiles.** One carrying `time` / `x-deep` /
`metrics[].role` / `availability` is lifted into a task on the way in and
produces a byte-identical SPEC, so no producer has to migrate on the same day
the compiler does. The compile report says `lifted_from_flat` when that
happened, which is the signal that something upstream has not moved yet.

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
Change the horizon, the season, the engine, and re-run: **every cell of the frame
must be identical.** That is the invariant the SPEC/run-config split exists to
protect, and this panel is how you check it rather than assume it.

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

- **Six questions, one schema.** Point at `interactions.csv` and ask, in turn:
  *forecast total line_total per item each month*, *recommend items to
  customers*, *classify each customer by status*, *segment customers into
  cohorts*, *rank the top 3 items for each customer*, *flag unusual spikes in
  monthly spend*. Nothing else changes. Every one picks its own kind from the
  sentence, and the frame contract is the same rule each time — keys, plus `t`
  when a key bins a clock, plus one column per aggregate.

- **Clustering, just by asking.** "segment customers into cohorts" produces it
  directly: no clock, no target, every metric carried alongside. It used to need
  a hand-edited SPEC, because `_finalize` demanded a `time.event` of every
  profile and the contract table forbade a clustering a target it never had.
  Note the SPEC is unchanged in shape from the supervised case: nothing in it
  ever said which column was a target.

- **A composite key.** Add a second entry to `keys[]`. The old `series_from` was
  a single string, so this was unrepresentable.

- **The leakage guard.** Add `"validity": {"arrival_from": "reported_dt"}` to the
  awards SPEC. The Sept 27 award was not reported until Oct 30, so as a
  `past_only` feature it is re-filed forward into Q4 — producing a Q4 row with a
  **null target and a live feature**. The counter reads
  `features_refiled_forward: 1`. That row existing at all is the guard and the
  `got_any` fix working together.

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
