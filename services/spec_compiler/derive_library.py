"""derive_library: the Mapping Layer's derive catalog.

A derive turns a raw cell into a usable value. Two scopes, because they run at
different times:

  * key scope   -- runs once per output row. Either bins a date into a bucket
                   (the `bin:*` family, which is how time stops being a special
                   position and becomes an ordinary grouping key), or projects a
                   bucket into a calendar scalar (week_of_year and friends).
  * row scope   -- runs once per input row, before aggregation, so an aggregate
                   can read it. No members are built yet; the catalog can still
                   describe them, and a SPEC that asks for one is told exactly
                   what is missing rather than getting a silent default.

Two return shapes:

  * bucket -- (label, sort_value). The label is what lands in the frame cell;
              the sort value orders the rows and feeds key-scope scalars.
  * scalar -- one number.

Same shape as function_library: the catalog is a sibling .csv when shipped and
an identical embedded copy otherwise, so a runtime-published service (which
ships only .py) still validates SPECs. Stdlib only.
"""

import csv
import io
import os
from datetime import date, timedelta

CATALOG_FILENAME = "derive_library.csv"

# Keep identical to derive_library.csv - the self_test asserts they match.
# `needs` is what the derive reads, and it is the whole basis of the one check
# that catches a clock pointed at a column of prose. A derive that needs a date
# is a promise about the DATA, not about the profile: no amount of correct
# authoring makes `bin:month` work over a column holding "shampoo".
# `ordered` is whether this derive's values form a sequence with a successor --
# whether "what comes next" is a question about the key at all. It is what a
# forecast actually requires: not time, but somewhere to project into.
#
# It is NOT the same as `returns: bucket`, and `year` is why. Both `year` and
# `month_of_year` return a scalar off the bucket, but 2024 is followed by 2025
# while December is followed by a January belonging to a different year. One can
# be forecast along and the other cannot, and this column is the only place that
# difference can be written down.
_EMBEDDED_CATALOG = """derive,scope,returns,needs,ordered,description
bin:day,key,bucket,date,yes,Bin a date into one calendar day
bin:week,key,bucket,date,yes,Bin a date into one ISO week
bin:month,key,bucket,date,yes,Bin a date into one calendar month
bin:quarter,key,bucket,date,yes,Bin a date into one calendar quarter
bin:year,key,bucket,date,yes,Bin a date into one calendar year
week_of_year,key,scalar,date,no,ISO week number of the bucket - cyclic
month_of_year,key,scalar,date,no,Calendar month number of the bucket - cyclic
quarter_of_year,key,scalar,date,no,Calendar quarter number of the bucket - cyclic
day_of_week,key,scalar,date,no,ISO weekday of the bucket (Monday is 1) - cyclic
year,key,scalar,date,yes,Calendar year of the bucket
row_number,key,position,,no,Each input row becomes its own frame row - position is not a sequence
"""

# The grains a bin: derive understands, in the order the old GRAINS tuple had.
GRAINS = ("day", "week", "month", "quarter", "year")


# ------------------------------------------------------------- bucket derives

def _bin(grain, d):
    """(label, sort_value) for one date at one grain.

    The sort value is the bucket's start date, so rows order correctly and any
    key-scope scalar derive has a real date to read. Labels are zero-padded so
    they also sort lexicographically, which keeps v1 frames byte-identical.
    """
    if grain == "day":
        return d.isoformat(), d
    if grain == "week":
        iso = d.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}", d - timedelta(days=d.weekday())
    if grain == "month":
        return f"{d.year}-{d.month:02d}", date(d.year, d.month, 1)
    if grain == "quarter":
        q = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{q}", date(d.year, 3 * (q - 1) + 1, 1)
    return str(d.year), date(d.year, 1, 1)


# ------------------------------------------------------------- scalar derives

def _scalar(name, d):
    if name == "week_of_year":
        return d.isocalendar()[1]
    if name == "month_of_year":
        return d.month
    if name == "quarter_of_year":
        return (d.month - 1) // 3 + 1
    if name == "day_of_week":
        return d.isoweekday()
    return d.year


_IMPLEMENTED = (
    {"bin:" + g for g in GRAINS}
    | {"week_of_year", "month_of_year", "quarter_of_year", "day_of_week", "year"}
    # `row_number` needs no function here: its value is the row's position, which
    # only the executor knows. It is listed as implemented because the mapping
    # layer handles the "position" scope directly in _key_tuple, the same way a
    # bin: spine is handled outside apply_bucket.
    | {"row_number"}
)


# ------------------------------------------------------------- catalog access

_catalog_cache = None


def _parse_catalog(text):
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        name = (r.get("derive") or "").strip()
        if not name:
            continue
        out[name] = {
            "scope": (r.get("scope") or "").strip(),
            "returns": (r.get("returns") or "").strip(),
            "needs": (r.get("needs") or "").strip(),
            "ordered": (r.get("ordered") or "").strip().lower() == "yes",
            "description": (r.get("description") or "").strip(),
        }
    return out


def embedded_catalog():
    """The catalog baked into this module (source of truth for the .csv)."""
    return _parse_catalog(_EMBEDDED_CATALOG)


def catalog():
    """The active catalog: the sibling .csv if present, else the embedded copy."""
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
    """True when a catalogued derive has working code here."""
    return name in _IMPLEMENTED


def resolve(name, what):
    """Three-state lookup: ready, catalogued-but-not-built, or unknown.

    Mirrors function_library's contract so an unmet SPEC always names the gap
    instead of falling back to a default.
    """
    name = str(name or "").strip()
    lib = catalog()
    entry = lib.get(name)
    if entry is None:
        raise ValueError(
            f"{what}: '{name}' is not in the derive library "
            f"(known: {', '.join(sorted(lib))}). Add it to "
            f"{CATALOG_FILENAME} and implement it in derive_library.py")
    if not is_implemented(name):
        raise ValueError(
            f"{what}: '{name}' is catalogued with {entry['scope']} scope but not yet "
            f"implemented - build it in derive_library.py before using it")
    return entry


def apply_bucket(name, d):
    """Run a bucket-returning derive. Returns (label, sort_value)."""
    if not name.startswith("bin:"):
        raise ValueError(f"derive '{name}' does not return a bucket")
    return _bin(name[4:], d)


def apply_scalar(name, d):
    """Run a scalar-returning derive over a bucket's sort value."""
    return _scalar(name, d)
