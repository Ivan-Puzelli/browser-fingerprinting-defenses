# SPDX-License-Identifier: MIT
"""Shared analysis library for the cross-browser fingerprinting-defense experiment.

One definition per quantity, imported by both the CLI scripts and the generated
notebook, so the two cannot drift apart.

    data       dataset loading, arm constants, canonical value extraction
    stability  the four repetition levels and per-component variance
    canvas     pixel-level canvas comparison
    entropy    the entropy budget and the anonymity-set capacity test
    similarity threshold matching over the full attribute vector
    linkage    cross-browser linkability
    figures    every figure, saved to figures/ and shown inline in the notebook

LaTeX export is deliberately not here: the paper reads the figures this package
writes, and nothing in the lab depends on the write-up existing.
"""
from . import canvas, data, entropy, figures, linkage, similarity, stability
from .data import ARMS, BASELINE, COLOR, DOMAINS, LABEL, load

__all__ = ["ARMS", "BASELINE", "COLOR", "DOMAINS", "LABEL", "load",
           "canvas", "data", "entropy", "figures", "linkage", "similarity",
           "stability"]
