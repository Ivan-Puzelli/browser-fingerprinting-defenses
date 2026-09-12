# SPDX-License-Identifier: MIT
"""Similarity matching, as opposed to exact matching.

FingerprintJS hashes the attribute vector, so a single farbled byte breaks the
join. An adversary is not obliged to do that. This module compares observations
by the *fraction of attributes on which they agree*, which is what the paper's
adaptive adversary does, and asks whether a threshold exists that links a
browser's own sessions while still separating it from every other arm.
"""
import collections
import itertools

import numpy as np

from .data import ARMS, FAMILY, attr_value, components, select
from .entropy import BITS, COMPOSITE


def attributes(rows):
    """The 43 attributes compared: every component, plus the header user agent.
    Composites are excluded so no surface is counted twice."""
    return sorted((set(components(rows)) | set(BITS)) - COMPOSITE)


def vectors(rows, level=3, attrs=None):
    """{arm: [observation vector, ...]} - one vector per Level-3 session."""
    attrs = attrs or attributes(rows)
    return {arm: [tuple(attr_value(r, a) for a in attrs)
                  for r in select(rows, arm, level)]
            for arm in ARMS}


def agreement(u, v):
    """Fixed-schema agreement; an incomplete vector has no comparable score."""
    if len(u) != len(v):
        raise ValueError("Attribute vectors must have the same length")
    if not u or any(a is None or b is None for a, b in zip(u, v)):
        return float("nan")
    return sum(a == b for a, b in zip(u, v)) / len(u)


def within(vecs, arm, max_pairs=None):
    """Agreement between independent sessions of the SAME arm."""
    pairs = list(itertools.combinations(vecs[arm], 2))
    if max_pairs:
        pairs = pairs[:max_pairs]
    return np.array([s for u, v in pairs if np.isfinite(s := agreement(u, v))])


def between(vecs, arm, other):
    """Agreement between sessions of two DIFFERENT arms."""
    return np.array([s for u in vecs[arm] for v in vecs[other]
                     if np.isfinite(s := agreement(u, v))])


def separation(rows, arm="brave", level=3):
    """Can a single threshold link one arm's sessions without also linking it to
    a different browser on the same host?

    Arms of the same product family are excluded from the comparison: Brave
    default and Brave aggressive are two settings of one browser, so their
    resemblance is not a false positive an adversary would have to avoid - it is
    the same browser twice. (They agree more closely with each other than Brave
    sessions do among themselves, which is the configuration-equivalence result
    reported separately.)

    Returns the within-arm distribution, the closest genuinely different
    browser, and the open interval of thresholds separating them - None if no
    such interval exists.
    """
    vecs = vectors(rows, level)
    w = within(vecs, arm)
    others = {o: between(vecs, arm, o) for o in ARMS
              if FAMILY[o] != FAMILY[arm]}
    if not w.size or any(not v.size for v in others.values()):
        raise ValueError("Separation requires complete observations for each compared arm")
    rival = max(others, key=lambda o: others[o].max())
    lo, hi = others[rival].max(), w.min()
    return dict(
        arm=arm, n_pairs=len(w), n_attributes=len(attributes(rows)),
        n_incomplete=sum(any(v is None for v in row) for vv in vecs.values() for row in vv),
        within_mean=float(w.mean()), within_min=float(w.min()),
        rival=rival, rival_max=float(others[rival].max()),
        window=(float(lo), float(hi)) if hi > lo else None,
        within=w, others=others,
        family_excluded=[o for o in ARMS if FAMILY[o] == FAMILY[arm] and o != arm])


def exact_matches(rows, arm, level=3, field="visitorId"):
    """How many of the same arm's sessions an exact hash would link: the size of
    the largest group sharing one identifier, against the number of sessions."""
    ids = [r[field] for r in select(rows, arm, level) if r.get(field)]
    if not ids:
        return 0, 0
    return collections.Counter(ids).most_common(1)[0][1], len(ids)
