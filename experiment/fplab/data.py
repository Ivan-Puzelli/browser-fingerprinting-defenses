# SPDX-License-Identifier: MIT
"""Dataset loading, arm constants and canonical value extraction.

Every consumer - the CLI scripts and the generated notebook alike - loads the
dataset through here, so the exclusions below are applied exactly once and
cannot drift between them.
"""
import json
import collections
import base64
import io
from functools import lru_cache
from pathlib import Path

# Arms in display order, each mapped to the undefended browser of its own
# engine. Judging a defense against its own engine means it is never charged
# for ordinary Chromium-vs-Gecko differences.
BASELINE = {"chrome": "chrome", "brave": "chrome", "brave-aggressive": "chrome",
            "firefox": "firefox", "firefox-rfp": "firefox", "tor-standard": "firefox"}
ARMS = list(BASELINE)

# Independent product families. brave and brave-aggressive are two settings of
# ONE product, so their agreement is not independent evidence of anything.
FAMILY = {"chrome": "chromium-stock", "brave": "brave", "brave-aggressive": "brave",
          "firefox": "gecko-stock", "firefox-rfp": "gecko-rfp", "tor-standard": "tor"}

LABEL = {"chrome": "Chrome", "brave": "Brave", "brave-aggressive": "Brave-aggr.",
         "firefox": "Firefox", "firefox-rfp": "Firefox RFP", "tor-standard": "Tor"}

COLOR = {"chrome": "#4C72B0", "brave": "#DD8452", "brave-aggressive": "#C44E52",
         "firefox": "#55A868", "firefox-rfp": "#8172B2", "tor-standard": "#7D4698"}

DOMAINS = ("site-a.test", "site-b.test")

# The no-JavaScript rows carry headers and no components, so they support no
# component-level statistic and appear in no result. Dropped on load.
DROP_BROWSERS = {"tor-nojs"}

# One run is excluded as contaminated rather than absorbed by a tolerance
# threshold: a threshold loose enough to swallow it (0.90) would also misread
# Brave's genuine low-rate screen-resolution noise (4/40 runs) as a fixed value.
# The exclusion is a single documented data-quality decision; a threshold would
# be a tunable knob that silently changes results.
#   firefox-rfp / L3 / run 10 reported Firefox/154.0 and languages en-US, while
#   the other 39 runs reported Firefox/147.0 and it-IT. Both anomalies are that
#   one run: a different browser build served the probe.
EXCLUDE = [dict(browser="firefox-rfp", level=3, run_index=10,
                reason="reported Firefox/154.0 + en-US; other 39 runs 147.0 + it-IT")]


def root(start=None):
    """Locate the experiment directory from anywhere at or below the repo."""
    here = Path(start or Path.cwd()).resolve()
    for p in (here, here / "experiment", *here.parents,
              Path(__file__).resolve().parent.parent):
        if (p / "data" / "results.jsonl").exists():
            return p
    raise FileNotFoundError(
        "data/results.jsonl not found - run `python run_experiment.py` first")


def excluded(r):
    return any(all(r.get(k) == v for k, v in e.items() if k != "reason")
               for e in EXCLUDE)


def load(path=None, verbose=True, return_report=False):
    """Load usable probe rows; preserve raw data and report rejected observations.

    Component-level gaps stay in the dataset and are handled per metric.
    Malformed JSON is an input error, never silently skipped.
    """
    source = Path(path) if path is not None else root() / "data" / "results.jsonl"
    rows, report = [], dict(total=0, non_js=0, excluded=0, failed=0, invalid=0)
    with source.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSON: {exc.msg}") from exc
            report["total"] += 1
            if not isinstance(r, dict):
                report["invalid"] += 1
            elif r.get("browser") in DROP_BROWSERS:
                report["non_js"] += 1
            elif r.get("error") is not None or r.get("error_stage") is not None:
                report["failed"] += 1
            elif (r.get("browser") not in ARMS or r.get("level") not in (2, 3, 4)
                  or r.get("domain") not in DOMAINS
                  or type(r.get("run_index")) is not int
                  or not isinstance(r.get("components", {}), dict)):
                report["invalid"] += 1
            elif excluded(r):
                report["excluded"] += 1
            else:
                rows.append(r)
    report["used"] = len(rows)
    if verbose:
        print("collection audit: " + ", ".join(f"{k}={v}" for k, v in report.items()))
    return (rows, report) if return_report else rows


@lru_cache(maxsize=128)
def valid_image(value):
    """A non-empty, decodable raw image, not a library verdict or placeholder."""
    from PIL import Image
    if not isinstance(value, str) or not value.startswith("data:image/"):
        return False
    try:
        header, encoded = value.split(",", 1)
        if ";base64" not in header:
            return False
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as im:
            im.verify()
        return True
    except (ValueError, OSError, SyntaxError):
        return False


def canvas_status(r):
    """Classify the L1 pair independently of upstream's skipped/unstable flag."""
    c = r.get("level1_canvas_call1")
    if r.get("error") is not None or (isinstance(c, dict) and "error" in c):
        return "failed"
    if not isinstance(c, dict):
        return "unavailable"
    return "valid" if all(isinstance(c.get(k), str) and valid_image(c[k])
                          for k in ("textImage1", "textImage2")) else "unavailable"


def value(r, comp):
    """Canonical string for one component reading.

    Uses only the component's `value`, dropping the `duration` field every
    component carries: that is a wall-clock timing measurement which changes on
    almost every run, and leaving it in would make every component look fully
    randomized.
    """
    v = (r.get("components") or {}).get(comp)
    if comp not in (r.get("components") or {}):
        return None
    if isinstance(v, dict):
        if "error" in v or ("value" not in v and "duration" not in v):
            return None
        # JSON omits a successful JS `undefined` value; FPJS retains duration.
        # This observed unsupported outcome is distinct from a missing/error entry.
        v = v.get("value")
    return json.dumps(v, sort_keys=True, default=str)


def attr_value(r, attr):
    """Canonical string for one *budget attribute*, which is not always a
    component. Three attributes are read from elsewhere:

    canvas        from the PATCHED raw image. `components.canvas` holds the
                  literal 'skipped' on Gecko and would score Tor and Firefox
                  RFP - the two most aggressive randomizers here - as
                  perfectly stable.
    userAgent     from the request header, which no in-page defense rewrites.
    supercookies  synthesised from the three storage probes.
    """
    if attr == "canvas":
        c = r.get("level1_canvas_call1")
        v = c.get("textImage1") if isinstance(c, dict) and "error" not in c else None
        return v if isinstance(v, str) and valid_image(v) else None
    if attr == "userAgent":
        return r.get("_user_agent_header")
    if attr == "supercookies":
        vals = [value(r, k) for k in ("localStorage", "sessionStorage", "indexedDB")]
        return json.dumps(vals) if all(v is not None for v in vals) else None
    return value(r, attr)


def select(rows, arm, level=None, domain=None, with_components=False):
    """Rows for one arm, optionally narrowed to a level and/or domain."""
    return [r for r in rows
            if r["browser"] == arm
            and (level is None or r["level"] == level)
            and (domain is None or r["domain"] == domain)
            and (not with_components or r.get("components"))]


def modal(rows, arm, comp, level=3):
    """The value an arm reports for a component, as its most common reading."""
    v = [v for r in select(rows, arm, level, with_components=True)
         if (v := attr_value(r, comp)) is not None]
    return collections.Counter(v).most_common(1)[0][0] if v else None


def components(rows):
    """Every component name present anywhere in the dataset."""
    return sorted({k for r in rows for k in (r.get("components") or {})})


def component_coverage(rows):
    """Counts per arm/component: observed outcomes, explicit errors and gaps.

    Successful undefined/null outcomes are observations, not collection failures.
    Raw canvas pairs have their own coverage counts in stability.level1().
    """
    import pandas as pd
    out = []
    for arm in ARMS:
        rr = select(rows, arm)
        for comp in components(rows):
            observed = failed = 0
            for r in rr:
                entry = (r.get("components") or {}).get(comp)
                failed += isinstance(entry, dict) and "error" in entry
                observed += value(r, comp) is not None
            out.append(dict(browser=arm, component=comp, n_total=len(rr),
                            n_observed=observed, n_failed=failed,
                            n_unavailable=len(rr)-observed-failed))
    return pd.DataFrame(out)
