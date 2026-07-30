"""function_library: the Mapping Layer's toolbox.

The SPEC names a function; this module is where the functions actually live.
Two kinds sit behind one catalog:

  * reductions (core shelf) collapse a bucket's list of measure values into one
    number: sum, mean, min, max, count, first, last. These are applied by the
    mapping layer's own `_agg`; the library only catalogs them.
  * group functions (domain shelves) need an entity column too — within a bucket
    they first total the measure per entity, then compute one number from those
    entity totals. `hhi` and `top_share` are the supply-chain shelf.

The catalog (name, shelf, kind, needs, description) is stored beside this module
as `function_library.csv`. It is loaded when present and otherwise falls back to
an identical embedded copy, so a runtime-published service (which ships only
`.py`) still validates SPECs. Implementations live here in Python; a function
that is catalogued but has no implementation is reported as "not yet built" when
a SPEC tries to use it — exactly the flowchart's "the library flags it so we
build it."

Stdlib only.
"""

import csv
import io
import os

CATALOG_FILENAME = "function_library.csv"

# The catalog, embedded verbatim so the service validates even when the .csv is
# not shipped alongside it. Keep this identical to function_library.csv — the
# mapping layer's self_test asserts they match.
_EMBEDDED_CATALOG = """function,shelf,kind,needs,description
sum,core,reduction,measure,Add a measure up
mean,core,reduction,measure,Average a measure
min,core,reduction,measure,Smallest value of a measure
max,core,reduction,measure,Largest value of a measure
count,core,reduction,,Count the rows in the bucket
first,core,reduction,measure,First value in the bucket by input order
last,core,reduction,measure,Last value in the bucket by input order
hhi,supply-chain,group,measure|entity,Herfindahl-Hirschman index: sum of squared percent shares of a measure across an entity
top_share,supply-chain,group,measure|entity,Largest single entity percent share of the measure in the bucket
"""

# Which catalogued functions actually have working code here. Reductions are run
# by the mapping layer's `_agg`; groups are run by `apply_group` below.
IMPLEMENTED_REDUCTIONS = {"sum", "mean", "min", "max", "count", "first", "last"}


# --------------------------------------------------------------- group functions

def _hhi(entity_sums):
    """Herfindahl-Hirschman index: sum of squared percentage shares.

    entity_sums maps entity -> net measure within one bucket. Shares are on the
    net total (so deobligations net out per entity, matching the concentration
    pipeline). A non-positive total has no defined shares -> None (undefined),
    never a fabricated number.
    """
    total = sum(entity_sums.values())
    if total <= 0:
        return None
    return sum((v / total * 100.0) ** 2 for v in entity_sums.values())


def _top_share(entity_sums):
    """Largest single entity's percentage share of the bucket total."""
    total = sum(entity_sums.values())
    if total <= 0:
        return None
    return max(entity_sums.values()) / total * 100.0


_GROUP_IMPL = {
    "hhi": _hhi,
    "top_share": _top_share,
}


# --------------------------------------------------------------- catalog access

_catalog_cache = None


def _parse_catalog(text):
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        name = (r.get("function") or "").strip()
        if not name:
            continue
        needs = [x.strip() for x in (r.get("needs") or "").split("|") if x.strip()]
        out[name] = {
            "shelf": (r.get("shelf") or "").strip(),
            "kind": (r.get("kind") or "").strip(),
            "needs": needs,
            "description": (r.get("description") or "").strip(),
        }
    return out


def embedded_catalog():
    """The catalog baked into this module (source of truth for the .csv)."""
    return _parse_catalog(_EMBEDDED_CATALOG)


def catalog():
    """The active catalog: the sibling .csv if present, else the embedded copy.

    Cached after first read. name -> {shelf, kind, needs, description}.
    """
    global _catalog_cache
    if _catalog_cache is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CATALOG_FILENAME)
        try:
            with open(path, encoding="utf-8") as f:
                _catalog_cache = _parse_catalog(f.read())
        except OSError:
            _catalog_cache = embedded_catalog()
    return _catalog_cache


def is_implemented(name):
    """True when a catalogued function has working code (reduction or group)."""
    return name in IMPLEMENTED_REDUCTIONS or name in _GROUP_IMPL


def apply_group(name, entity_sums):
    """Run a group function over {entity: net measure}. None when undefined/empty."""
    if not entity_sums:
        return None
    fn = _GROUP_IMPL.get(name)
    if fn is None:  # pragma: no cover - guarded by is_implemented upstream
        raise ValueError(f"group function '{name}' has no implementation")
    return fn(entity_sums)
