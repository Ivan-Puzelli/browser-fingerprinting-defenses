# SPDX-License-Identifier: MIT
"""Stability of the fingerprint across the four nested repetition levels.

L1  two canvas reads inside one script execution
L2  repeated reads inside one browser session
L3  independent sessions, fresh process and profile each
L4  two origins visited inside one session
"""
import collections

import numpy as np

from .data import ARMS, BASELINE, DOMAINS, attr_value, canvas_status, components, modal, select, value


def level1(rows):
    """Per-read determinism: do two back-to-back canvas draws agree?"""
    import pandas as pd
    out = []
    for arm in ARMS:
        rr = select(rows, arm)
        counts = collections.Counter(canvas_status(r) for r in rr)
        calls = [r["level1_canvas_call1"] for r in rr if canvas_status(r) == "valid"]
        intra = [c["textImage1"] == c["textImage2"] for c in calls]
        out.append(dict(browser=arm, n=len(rr), n_valid=counts["valid"],
                        n_failed=counts["failed"], n_unavailable=counts["unavailable"],
                        intra_call_match=float(np.mean(intra)) if intra else np.nan,
                        upstream=dict(collections.Counter(
                            c.get("upstreamVerdict") for c in calls))))
    return pd.DataFrame(out)


def stability(rows, level, field="visitorId"):
    """Share of runs sharing the modal value, computed WITHIN each domain and
    then averaged.

    Grouping by domain is essential: Brave's farbling key includes domain_key,
    so pooling site-a and site-b would guarantee two distinct values and halve
    the score for a reason that is Level 4's subject, not Level 2's or 3's.
    """
    out = {}
    for arm in ARMS:
        per_domain = []
        for dom in DOMAINS:
            vals = [r[field] for r in select(rows, arm, level, dom) if r.get(field)]
            if vals:
                per_domain.append(
                    collections.Counter(vals).most_common(1)[0][1] / len(vals))
        out[arm] = float(np.mean(per_domain)) if per_domain else np.nan
    return out


def _by_run(rows, arm, level, need_components=False):
    byrun = collections.defaultdict(dict)
    for r in select(rows, arm, level, with_components=need_components):
        byrun[r["run_index"]][r["domain"]] = r
    return [v for v in byrun.values() if all(d in v for d in DOMAINS)]


def level4_same(rows):
    """Fraction of same-session domain pairs returning an identical fingerprint."""
    return {arm: (float(np.mean([len({r["visitorId"] for r in v.values()}) == 1
                                 for v in pairs]))
                  if (pairs := [v for v in _by_run(rows, arm, 4)
                                if all(r.get("visitorId") for r in v.values())]) else np.nan)
            for arm in ARMS}


def by_level(rows):
    """The headline stability table: one row per arm, one column per level."""
    import pandas as pd
    return pd.DataFrame({
        "L1 canvas":  level1(rows).set_index("browser").intra_call_match,
        "L2":         pd.Series(stability(rows, 2)),
        "L3":         pd.Series(stability(rows, 3)),
        "L4 same-fp": pd.Series(level4_same(rows)),
    }).loc[ARMS].round(3)


def cross_domain(rows):
    """Per-component agreement between the two domains within one session."""
    rec = {}
    for arm in ARMS:
        pairs = _by_run(rows, arm, 4, need_components=True)
        if not pairs:
            continue
        comps = sorted({k for v in pairs for r in v.values() for k in r["components"]})
        readings = {c: [(attr_value(v[DOMAINS[0]], c), attr_value(v[DOMAINS[1]], c))
                        for v in pairs] for c in comps}
        valid = {c: [(a, b) for a, b in vv if a is not None and b is not None]
                 for c, vv in readings.items()}
        agree = {c: float(np.mean([a == b for a, b in vv]))
                 for c, vv in valid.items() if vv}
        rec[arm] = dict(n_pairs=len(pairs),
                        n_valid_comparisons=sum(map(len, valid.values())),
                        n_unavailable_comparisons=len(pairs)*len(comps)-sum(map(len, valid.values())),
                        component_agreement=float(np.mean(list(agree.values()))) if agree else np.nan,
                        differing=[c for c, a in agree.items() if a < 1.0])
    return rec


def cross_domain_frame(rows):
    """`cross_domain` formatted for display."""
    import pandas as pd
    xd = cross_domain(rows)
    return pd.DataFrame({a: {"pairs": d["n_pairs"],
                             "component agreement": round(d["component_agreement"], 3),
                             "components differing": ", ".join(d["differing"]) or "none"}
                         for a, d in xd.items()}).T


# ---------------------------------------------------------- component variance
LEAKED, PINNED, RANDOMIZED = 0, 1, 2


def variance(rows, level=3):
    """Distinct values per (component, arm) at one level."""
    import pandas as pd
    comps = components(rows)
    return pd.DataFrame({
        arm: {c: len({v for r in select(rows, arm, level, with_components=True)
                      if (v := attr_value(r, c)) is not None})
              for c in comps} for arm in ARMS})


def dispositions(rows, var=None, level=3):
    """Three coarse states per (component, arm), restricted to components that
    carry signal. A component stable everywhere AND identical in every arm tells
    the reader nothing, and about half of them are exactly that.

    This is the *coarse* view used for the heatmap. The entropy budget uses the
    stricter classifier in `fplab.entropy`, which additionally requires positive
    evidence before crediting a pinned value.
    """
    import pandas as pd
    var = variance(rows, level) if var is None else var

    def state(comp, arm):
        if var.loc[comp, arm] == 0:
            return 3  # unavailable, not evidence of stability or a defense
        if var.loc[comp, arm] > 1:
            return RANDOMIZED
        return (PINNED if modal(rows, arm, comp, level) != modal(rows, BASELINE[arm], comp, level)
                else LEAKED)

    informative = [c for c in components(rows)
                   if (var.loc[c] > 1).any()
                   or len({modal(rows, a, c, level) for a in ARMS}) > 1]
    S = pd.DataFrame({a: {c: state(c, a) for c in informative} for a in ARMS})
    return S.loc[informative]
