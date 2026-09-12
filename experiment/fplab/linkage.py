# SPDX-License-Identifier: MIT
"""Cross-browser linkability on one machine.

Asks whether a tracker could join the same user across two different browsers,
using only the components that farbling does not touch.
"""
from .data import ARMS, components, modal

# The surfaces Brave farbles. Excluded here because their instability is the
# subject of the earlier sections, not of this one.
FARBLED = ("canvas", "audio", "plugins", "hardwareConcurrency", "screenResolution")


def immune_components(rows):
    """Non-farbled components that actually discriminate between arms.

    A component identical in every arm carries no information and would pad
    every pairwise score toward 1.0, so it is excluded.
    """
    return [c for c in components(rows)
            if c not in FARBLED and len({modal(rows, a, c) for a in ARMS}) > 1]


def constant_components(rows):
    """Non-farbled components identical in every arm - the ones excluded above."""
    return [c for c in components(rows)
            if c not in FARBLED and len({modal(rows, a, c) for a in ARMS}) == 1]


def immune_matrix(rows):
    """-> (agreement matrix over immune components, the component list)."""
    import numpy as np
    import pandas as pd
    immune = immune_components(rows)
    profile = {a: {c: modal(rows, a, c) for c in immune} for a in ARMS}
    M = pd.DataFrame({a: {b: float(np.mean([profile[a][c] == profile[b][c]
                                            for c in immune if profile[a][c] is not None
                                            and profile[b][c] is not None]))
                          for b in ARMS} for a in ARMS}).loc[ARMS, ARMS]
    return M, immune
