# SPDX-License-Identifier: MIT
"""Pixel-level comparison of canvas defenses.

Decodes the raw canvases preserved by Patch 1 and measures, per arm, how many
pixels change between independent sessions and by how much. This separates a
sparse +/-1 mask (Brave) from large readout changes (Gecko RFP and Tor),
which stability rates alone cannot.
"""
import base64
import collections
import io
from typing import NamedTuple

import numpy as np
from PIL import Image

from .data import ARMS, DOMAINS, select

MAX_PAIRS = 20


class Diff(NamedTuple):
    """One image-to-image comparison."""
    frac: float                    # fraction of pixels differing, 0..1
    max_delta: int                 # largest per-channel change, 0..255
    hist: collections.Counter      # distribution of non-zero channel deltas

    @property
    def pct(self):
        return 100 * self.frac


def decode(r, key="textImage1", call="level1_canvas_call1"):
    """One canvas from one row, as an RGBA integer array, or None."""
    c = r.get(call)
    v = c.get(key) if isinstance(c, dict) and "error" not in c else None
    if not v or not str(v).startswith("data:image"):
        return None
    try:
        return np.asarray(Image.open(io.BytesIO(
            base64.b64decode(v.split(",", 1)[1], validate=True))).convert("RGBA")).astype(int)
    except (ValueError, IndexError, OSError, SyntaxError):
        return None


def compare(a, b):
    """Diff between two decoded canvases, or None if either is unusable."""
    if a is None or b is None or a.shape != b.shape:
        return None
    d = np.abs(a - b)
    nz = d[d > 0]
    changed = d.sum(axis=2) > 0
    return Diff(changed.sum() / changed.size,
                int(nz.max()) if nz.size else 0,
                collections.Counter(nz.tolist()))


def _pairs_against_first(imgs, max_pairs=MAX_PAIRS):
    """Compare one reference image against each of the next `max_pairs`."""
    if len(imgs) < 2:
        return []
    return [d for d in (compare(imgs[0], imgs[i])
                        for i in range(1, min(len(imgs), max_pairs + 1))) if d]


def mechanism(rows, level=3, max_pairs=MAX_PAIRS):
    """Cross-session divergence, balanced across domains, never across origins.

    Each domain uses its first usable image as reference. These comparisons
    share references and are descriptive, not independent replicate pairs.
    """
    out = {}
    for arm in ARMS:
        diffs = []
        for j, domain in enumerate(DOMAINS):
            imgs = [i for r in select(rows, arm, level, domain)
                    if (i := decode(r)) is not None]
            limit = max_pairs // len(DOMAINS) + (j < max_pairs % len(DOMAINS))
            diffs.extend(_pairs_against_first(imgs, limit))
        if not diffs:
            continue
        hist = collections.Counter()
        for d in diffs:
            hist.update(d.hist)
        mx = max(d.max_delta for d in diffs)
        out[arm] = dict(frac_changed=float(np.mean([d.frac for d in diffs])),
                        pct_changed=100 * float(np.mean([d.frac for d in diffs])),
                        max_delta=mx, only_pm1=(mx == 1),
                        n_pairs=len(diffs), hist=hist)
    return out


def mechanism_frame(rows, level=3, max_pairs=MAX_PAIRS):
    """`mechanism` as a DataFrame of the three reported columns."""
    import pandas as pd
    m = mechanism(rows, level, max_pairs)
    return pd.DataFrame({a: {k: m[a][k] for k in ("pct_changed", "max_delta", "n_pairs")}
                         for a in m}).T.rename(columns={"n_pairs": "pairs"})


def _by_run(rows, arm, level):
    """{run_index: {domain: row}} for one arm at one level."""
    byrun = collections.defaultdict(dict)
    for r in select(rows, arm, level):
        byrun[r["run_index"]][r["domain"]] = r
    return byrun


def properties(rows, arms=("brave", "brave-aggressive")):
    """Brave's three farbling properties, measured on pixels rather than hashes.

    P3  deterministic within a session   - two Level-2 reloads, same domain
    P1  site-specific                    - the two domains, one Level-4 session
    P2  session-specific                 - independent Level-3 sessions
    """
    mech = mechanism(rows, 3)
    out = {}
    for arm in arms:
        r2 = select(rows, arm, 2, domain=DOMAINS[0])
        within = compare(decode(r2[0]), decode(r2[1])) if len(r2) > 1 else None
        cross = [d for d in (compare(decode(v[DOMAINS[0]]), decode(v[DOMAINS[1]]))
                             for v in _by_run(rows, arm, 4).values() if len(v) == 2) if d]
        m = mech.get(arm, {})
        out[arm] = dict(
            p3_within_session=within.frac if within else None,
            p1_cross_domain=float(np.mean([d.frac for d in cross])) if cross else None,
            p1_max_delta=max((d.max_delta for d in cross), default=None),
            p2_cross_session=m.get("frac_changed"),
            p2_max_delta=m.get("max_delta"))
    return out


def properties_frame(rows, arms=("brave", "brave-aggressive")):
    """`properties` formatted for display, one column per arm."""
    import pandas as pd
    p = properties(rows, arms)
    def fmt(v):
        return "unavailable" if v is None else f"{100*v:.2f}% changed"
    return pd.DataFrame({a: {
        "P3 within session (reloads)":   fmt(p[a]['p3_within_session']),
        "P1 across domains (1 session)": fmt(p[a]['p1_cross_domain']),
        "P2 across sessions":           fmt(p[a]['p2_cross_session']),
    } for a in arms})
