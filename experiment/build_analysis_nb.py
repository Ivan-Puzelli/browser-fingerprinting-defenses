# SPDX-License-Identifier: MIT
"""Generate analysis.ipynb - a plug-and-play notebook regenerating every table
and figure in the paper.

Every quantity is computed by the `fplab` package, which the CLI scripts import
too, so the notebook and the scripts cannot report different numbers. Cells here
are short on purpose: they select, display and comment, they do not re-implement.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t.strip()))
def co(t): C.append(nbf.v4.new_code_cell(t.strip()))

md("""
# Fingerprinting Defense Experiment — Analysis

Regenerates every table and figure in the paper from `data/results.jsonl`.

**Run order:** top to bottom. No server needed — this reads collected data only.
To re-collect, run `python run_experiment.py` first.

All computation lives in the `fplab` package next to this notebook, shared with
`entropy_budget.py` and `canvas_mechanism.py`. The cells below select, display and
interpret; none of them re-implement anything.

| Section | Produces |
|---|---|
| 1 | Level-1 per-read determinism |
| 2 | Stability by level (the three farbling properties) |
| 3 | Canvas mechanism at pixel level |
| 4 | Per-component breakdown |
| 5 | Entropy budget, robustness and anonymity-set capacity |
| 6 | The adaptive adversary: dropped channels and similarity matching |
| 7 | Cross-browser linkability |
| 8 | What was generated |

Every figure is written to `figures/` **and** rendered inline, so the notebook shows
exactly the image the paper uses.
""")

md("## 0 — Setup")
co("""
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

# Make `fplab` importable whether the kernel starts in experiment/, at the repo
# root, or anywhere below.
for p in (Path.cwd(), Path.cwd() / "experiment", *Path.cwd().parents):
    if (p / "fplab" / "__init__.py").exists():
        sys.path.insert(0, str(p)); break

import numpy as np, pandas as pd
import fplab
from fplab import (data, stability as st, canvas, entropy as E, linkage,
                   similarity as sim, figures)
from fplab.data import ARMS, BASELINE, COLOR

pd.set_option("display.width", 120)
print(f"reading from {data.root()}")
""")
co("""
rows, collection_audit = data.load(return_report=True)
display(pd.Series(collection_audit, name="observations"))
""")
co("""
# Coverage - confirm the design was actually collected before trusting any number.
df = pd.DataFrame([{k: r.get(k) for k in
                    ("browser", "domain", "level", "run_index", "visitorId")} for r in rows])
display(df.pivot_table(index="browser", columns="level", values="run_index",
                       aggfunc="count", fill_value=0))
""")

md("""
## 1 — Level 1: determinism within one script execution

The canvas is drawn twice back to back. If the two reads differ, the browser
noises *every* read; if they match, any instability must come from a coarser
grain (new session, new site).

Only valid, decodable image pairs enter the rate. Failed and unavailable pairs
are counted separately; an empty denominator is reported as unavailable, not 1.00.
""")
co("""
t_l1 = st.level1(rows); display(t_l1)
""")
co("""
# Read: 1.00 = the two draws are byte-identical (deterministic per read).
#       0.00 = every read returns a different image.
for _, r in t_l1.iterrows():
    kind = ("deterministic per read" if r.intra_call_match == 1
            else "noised on every read" if r.intra_call_match == 0 else "mixed")
    print(f"{r.browser:18} {r.intra_call_match:.2f}  {kind}")
""")

md("""
### What Level 1 shows

Two canvas draws inside a single script execution.

- **Chrome, Firefox, Brave — `1.00`.** The two images are byte-identical: no defense is
  acting *per read*.
- **Firefox RFP, Tor — `0.00`.** The two draws differ *within one page load*. Gecko
  re-randomizes on every single read.

This is the distinction the unpatched library cannot make, and the whole reason for patch
P1. FingerprintJS reads the difference between two draws as a signal to abandon the canvas
on Gecko — so it reports the most aggressively randomized arms here as perfectly stable.

Brave scoring `1.00` is not a failure of farbling. Its key is derived per session and
origin, so it is deterministic *within* a page by design. Levels 3 and 4 are where that
determinism is broken.
""")

md("""
## 2 — Levels 2, 3, 4: the three farbling properties

Each level holds one more thing constant, so a drop between two columns names
exactly which repetition the defense reacts to.
""")
co("""
t_lv = st.by_level(rows); display(t_lv)
""")
co("""
figures.stability_by_level(t_lv)
""")

md("""
### 2b — Level 4: what actually differs across domains

The `visitorId` is a hash, so it tells us only *whether* two sites saw the same
fingerprint. This breaks it down per component: which surfaces carry the
site-specificity, and how much of the vector is shared regardless.
""")
co("""
xd = st.cross_domain(rows)
display(st.cross_domain_frame(rows))
""")
co("""
figures.cross_domain_similarity(xd)
""")
co("""
# A difference can reflect origin-keying OR per-read randomization.
for a, d in xd.items():
    tag = "differs across the paired reads" if d["differing"] else "identical to both sites"
    print(f"{a:18} agreement {d['component_agreement']:.3f}  {tag}")
""")

md("""
### What Levels 2-4 show

One row per configuration; each column holds one more thing constant.

- **L2 is `1.00` for everything.** No defense changes the fingerprint between reloads of a
  live session, Brave included: its farbling key lives in RAM for the life of the session.
- **Only Brave drops at L3** (`0.05 = 1/20`) — twenty fresh processes produced twenty
  distinct identifiers. Session-specificity, farbling property P2.
- **Only Brave drops at L4** (`0.00`) — the two origins never returned the same identifier
  within one session. Site-specificity, P1.

**The result is in the flat rows, not in Brave's.** Tor scores `1.00` everywhere, exactly
like undefended Chrome. Uniformity does not make a fingerprint unlinkable; it makes it
*shared*. It defeats tracking by making the value useless for singling anyone out, not by
making it vary. The two philosophies are not stronger and weaker versions of one defense —
they break different halves of the trackability condition, and this is where that becomes
visible.

Mean component agreement is 0.905 for Brave and 0.976 for RFP/Tor when raw canvas
is included. In RFP/Tor, per-read canvas changes do not establish origin-keying.
""")

md("""
## 3 — Canvas mechanism, measured on pixels

Stability says *whether* the canvas changed. Decoding the raw images preserved
by Patch 1 says *how* — which separates Brave's sparse ±1 mask from Gecko's
readout replacement. Each arm uses 20 same-domain comparisons: ten against the first
usable image of each domain. Shared references make these descriptive comparisons,
not 20 independent experimental replicates.
""")
co("""
t_mech = canvas.mechanism_frame(rows); display(t_mech)
""")
co("""
# max_delta == 1 means every altered channel moved by exactly one unit:
# the signature of a sparse +/-1 mask rather than readout replacement.
for arm, r in t_mech.iterrows():
    kind = ("no change" if r.pct_changed == 0
            else "sparse +/-1 mask" if r.max_delta == 1 else "large readout changes")
    print(f"{arm:18} {r.pct_changed:6.2f}% of pixels, max delta {int(r.max_delta):3}  {kind}")
""")
co("""
figures.canvas_mechanism(t_mech)
""")
md("""
`max_delta` compresses the whole distribution into one number. The figure below opens it
back up: for each browser that alters the canvas at all, how its altered colour channels
are distributed over the 0-255 range of possible changes. The two families do
categorically different things, so they get a panel each and share a log y axis.
""")
co("""
figures.canvas_delta_spectrum(canvas.mechanism(rows))
""")
co("""
# The three properties, verified on pixels rather than hashes (Brave only).
display(canvas.properties_frame(rows))
""")

md("""
### What the pixels show

Stability rates say *whether* a canvas changed. Only the images say *how*.

- **Brave** alters roughly 3.5% of pixels, and across every pair each altered channel moves
  by exactly `1` — a sparse plus/minus-one mask applied at the API boundary. Measured here,
  not cited.
- **Firefox RFP and Tor** alter 100% of pixels, with maximum channel deltas of 252 and
  236 in these comparisons. This describes exported pixels, not the internal rendering.
- **Chrome and plain Firefox** alter nothing.

`max_delta` is the discriminator. Two defenses that look identical in a stability table are
doing structurally different things, and the mechanism is what predicts how each behaves
once a tracker adapts to it.
""")

md("""
## 4 — Which components vary across independent sessions
""")
co("""
COMPS = data.components(rows)
print(f"{len(COMPS)} components collected")
coverage = data.component_coverage(rows)
gaps = coverage[(coverage.n_failed > 0) | (coverage.n_unavailable > 0)]
display(gaps if len(gaps) else "No failed or missing component observations")
""")
co("""
# Distinct values per (component, arm) at Level 3, keeping only those that move
# somewhere. The 'duration' field every component carries is a wall-clock timing
# measurement and is stripped by data.value() - leaving it in would make every
# component look fully randomized.
var = st.variance(rows)
display(var[(var > 1).any(axis=1)])
""")
co("""
# Coarse three-state view, restricted to components that carry signal. A
# component stable everywhere AND identical in every arm tells the reader
# nothing, and about half of the 42 are exactly that.
S = st.dispositions(rows, var)
print(f"{len(S)} informative components of {len(COMPS)}")
""")
co("""
figures.component_heatmap(S)
""")

md("""
### Reading the disposition table

The heatmap reports observations against the baseline of the same engine:
**same as baseline**, **different but stable**, **varies across sessions**, or
**unavailable**. Canvas uses raw images, not the library's skipped/unstable labels.

A stable difference is not by itself evidence of population-wide pinning: it can
also reflect version skew. The entropy budget below applies its own stricter rules.
Components constant and identical in all six arms are omitted from this display;
this does not establish zero entropy across a population of different machines.
""")

md("""
## 5 — Entropy budget

Weight each attribute by its published population entropy, then ask whether a
tracker can still use it. An attribute pays out only if it is **present**,
**stable** across sessions, and **genuine** (equal to the real machine value).

> Population entropy cannot be measured from one machine. These weights are
> *citations*, not results — see `fplab.entropy.BITS` for every source, the two
> approximate mappings, and the reason each frozen attribute is dropped.
""")
co("""
# Every weight, its source, and whether it still applies to a 2026 browser.
display(pd.DataFrame(E.BITS).T[["pano", "amiu", "hiding", "berke", "frozen"]])
""")
co("""
WEIGHTING = "amiu"
table, totals = E.budget(rows, WEIGHTING)
attrs = E.weighted_attrs(WEIGHTING)

out = pd.DataFrame({arm: {a: table[(arm, a)][0] for a in attrs} for arm in ARMS})
out.insert(0, "bits", pd.Series({a: E.BITS[a][WEIGHTING] for a in attrs}).round(2))
display(out)

surviving = pd.Series({a: totals[a][0] for a in ARMS})
reduction = pd.Series(E.reduction(totals))
print("\\nsurviving bits:"); print(surviving.round(1).to_string())
print("\\nreduction vs same-engine baseline (%):"); print(reduction.round(0).to_string())
""")
co("""
figures.entropy_budget(table, totals, attrs, WEIGHTING)
""")

md("""### Does the answer depend on the price list?

The dispositions above are measured; only the bits attached to them are borrowed. So hold
the measurement fixed and vary the weighting. The last row borrows nothing at all: it
counts how many of the 43 collected attributes still leak, weighting `colorDepth` and
`canvas` alike.""")
co("""
rob = pd.DataFrame({E.STUDY_NAME[w]: E.reduction(E.budget(rows, w)[1])
                    for w in E.WEIGHTINGS}).T
rob.insert(0, "N", [E.STUDY_N[w] for w in E.WEIGHTINGS])

leaked, unw, n_attr = E.unweighted_reduction(rows)
rob.loc[f"Unweighted, {n_attr} attributes"] = ["---", *[unw[a] for a in ARMS]]
display(rob.round(0))
print(f"\\nattributes still leaking, of {n_attr}:")
print(pd.Series(leaked).to_string())
""")
co("""
figures.robustness(rob)
""")

md("""### Does the answer depend on classification assumptions?

Screen-size quantization and Tor's font allowlist can retain host differences.
The following alternatives charge their full published weights instead of granting
zero residual bits. They bracket model assumptions, not measured population entropy.
Tor stays first here; RFP and Brave swap order when the screen credit is withdrawn.

Only the two Gecko arms hold either credit, so only they can move: Brave's column is
constant *by construction*, not because its score is robust to these assumptions. It
stays in the table as the fixed reference the other two are read against, and is left
out of the figure, where four identical points would assert a robustness never tested.
""")
co("""
sensitivity = E.classification_sensitivity(rows, WEIGHTING)
display(sensitivity[["brave", "firefox-rfp", "tor-standard"]].round(2))
figures.classification_sensitivity(sensitivity)
""")
co("""
# How much the stale 2010 weights inflate the baseline.
for w in E.WEIGHTINGS:
    naive = E.budget(rows, w, drop_frozen=False)[1]["chrome"][0]
    corr = E.budget(rows, w, drop_frozen=True)[1]["chrome"][0]
    if naive:
        print(f"{w:6} chrome: {naive:.1f} naive vs {corr:.1f} corrected "
              f"-> {100*(naive-corr)/naive:.0f}% overstated")
""")

md("""### Anonymity-set capacity

The reduction percentages say how much entropy a defense removed. They cannot say
whether it removed *enough*. Lying moves a user into the crowd of everyone telling
the same lie; for the surviving bits not to single them out inside that crowd, the
crowd must be larger than `2**(surviving bits)`.

So compare the residual against the browser's own user base:

$$k = \\\\frac{N}{2^{H_{res}/\\\\delta}}$$

`k` is a **modelled crowd size** under an assumed effective fingerprint space, not
a measured anonymity set. A value below one does not prove individual identification.
`delta` reduces the bit total to allow for attribute correlation; 1 and 2 are
illustrative scenarios, not confidence bounds.""")
co("""
cap = E.capacity(rows, WEIGHTING)
capdf = pd.DataFrame({arm: {"N (users)": cap[arm]["population"],
                            "H_res (bits)": round(cap[arm]["bits"], 2),
                            "k (delta=1)": cap[arm]["k"][1.0],
                            "k (delta=2)": cap[arm]["k"][2.0]} for arm in ARMS}).T
display(capdf.style.format({"N (users)": "{:.1e}",
                            "k (delta=1)": "{:.2e}", "k (delta=2)": "{:.2e}"},
                           na_rep="---"))

print()
for arm in ARMS:
    c = cap[arm]
    if c["population"] is None:
        print(f"{arm:18} {c['bits']:6.2f} bits - no separate population "
              f"(RFP users are Firefox users)")
        continue
    k1, k2 = c["k"][1.0], c["k"][2.0]
    verdict = ("k < 1 at both" if k2 < 1 else
               "k < 1 only at d=1" if k1 < 1 else "k >= 1 at both")
    print(f"{arm:18} {c['bits']:6.2f} bits -> k={k1:9.2e} (d=1), {k2:9.2e} (d=2)  {verdict}")
""")
co("""
figures.anonymity_sets(cap)
""")

md("""
### Reading the budget, and then the capacity

`bits` is borrowed from the literature; every disposition beside it is measured here. An
attribute pays out its published entropy only if it is present, stable and genuine — fail
one and it contributes zero.

Under the default classification, Tor removes about 90% of the weighted surface,
Firefox RFP about 41%, Brave about 32%. This ordering holds across the tested weightings,
but the RFP/Brave ordering depends on the screen-classification assumption above.

**The capacity table is where the percentages stop mattering.** Chrome and Firefox carry
*identical* residual bits, yet their anonymity sets differ by more than an order of
magnitude — same score, different assumed crowd. RFP's user count is unavailable,
so no crowd-size verdict is assigned to it. Tor alone has modelled k >= 1 at delta=1.
""")

md("""
## 6 — The adaptive adversary

Everything above assumes a tracker that reads every channel and matches by exact
equality — which is what FingerprintJS does, and what the `visitorId` is. A tracker
is not obliged to do either. This section drops both assumptions in turn.

### 6a — Discarding the noisy channels

A tracker that knows a channel is noised can simply stop reading it. That costs it the
channel's bits, but it costs the defense everything the channel was contributing. The
budget is recomputed with the channel removed from **both** sides.

Of the five channels Brave farbles, only three carry a price: `audio` has no published
weight and `plugins` is frozen out of the budget, so neither gets a column — a scenario
that cannot move is not evidence that the channel is worthless. `-all farbled` is
therefore canvas + hardwareConcurrency + screenResolution.
""")
co("""
sweep = E.drop_channel_sweep(rows)
display(pd.DataFrame(sweep).round(2))
""")
co("""
figures.drop_channels(sweep)
""")
co("""
# The extreme case: the tracker discards every channel Brave farbles.
bits, red, base = E.drop_channels(rows, drop=E.FARBLED)
print(f"undefended baseline falls to {base:.2f} bits")
for arm in ARMS:
    print(f"  {arm:18} {bits[arm]:6.2f} bits   reduction {red[arm]:5.1f}%")
""")

md("""
**Brave's benefit goes to exactly zero.** Not approximately — the residual and the
baseline are the same number, because every attribute Brave still leaks is one Chrome
leaks too. Tor barely moves: almost none of its saving ever came from the noisy channels,
so discarding them costs the tracker more than it costs Tor.

For Firefox RFP, the raw canvas is the changing channel. Its user population is not
measured here, so this analysis alone cannot determine its real-world anonymity set.
""")

md("""
### 6b — Matching by similarity instead of by equality

The other assumption is exact matching. Compare two observations instead by the
**fraction of the 43 attributes on which they agree**, and ask whether a single threshold
links a browser's own sessions without also linking it to a different browser.
""")
co("""
sep = sim.separation(rows, "brave")
n_same, n_tot = sim.exact_matches(rows, "brave")
print(f"exact hashing links at most {n_same} of {n_tot} Brave sessions")
print(f"similarity: {sep['n_pairs']} session pairs, agreement "
      f"mean {sep['within_mean']:.3f}, min {sep['within_min']:.3f}")
print(f"closest different browser: {sep['rival']} at {sep['rival_max']:.3f}")
if sep["window"]:
    lo, hi = sep["window"]
    threshold = (lo + hi) / 2
    print(f"-> a threshold of {threshold:.6f} separates the observed groups")
print(f"incomplete observations excluded from similarity: {sep['n_incomplete']}")
print(f"(same-family arms excluded from the comparison: {sep['family_excluded']})")
""")
co("""
figures.similarity_separation(sep)
""")

md("""
**Farbling's unlinkability is a property of the matching function, not of the
fingerprint.** Exact hashing links none of the 780 Brave session pairs — forty runs, forty
distinct identifiers. A threshold of 0.87 links all of them and none of the compared
other-product observations on this host. The 780 pairs reuse the same 40 observations;
they are not independent trials and do not estimate population tracking precision.

The limit of the demonstration, stated precisely: the separation is from a *different
browser on the same host*, not from the *same browser on a different host*, which one
machine cannot show. What the single machine does establish is the direction — the
attributes on which two Brave sessions agree are exactly the ones describing the host
rather than the session.
""")

md("""
## 7 — Cross-browser linkability

Agreement on farbling-immune components across configurations on one machine.
Describes resemblance across browsers; it does not estimate tracking success.
""")
co("""
M, immune = linkage.immune_matrix(rows)
print(f"{len(immune)} discriminating farbling-immune components of {len(COMPS)}")
display(M.round(2))
""")
co("""
figures.cross_browser_linkability(M)
""")
co("""
const = linkage.constant_components(rows)
print(f"{len(const)} components identical in every arm (excluded as "
      f"non-discriminating): {const[:6]} ...")
print("-> the matrix clusters by ENGINE FAMILY, not by machine.")
""")

md("""
### Reading the linkability matrix

Agreement is computed only over components that farbling does not touch *and* that differ
between at least two arms; constants would pad every cell toward 1.0.

The matrix clusters by **engine family**, not by machine. Two browsers on this one host are
not automatically linkable: what links them is a shared engine, and what separates them is
what the defense pins. This is the cross-browser scope defined in the paper — the strongest
of the three scopes, and the one no first-party cookie can reach.
""")

md("""
## 8 — What this notebook generated

Everything above wrote its figure to `figures/` as it ran. That directory is the
notebook's only output: the analysis is reproducible from `data/results.jsonl`
alone, and nothing here needs the write-up to exist.

The paper reads four of these figures, plus the two appendix captures, straight
from `figures/`. It imports no tables: every number in it is typed from the
tables printed above.
""")
co("""
FIG = data.root() / "figures"
for f in sorted(FIG.glob("*.png")):
    print(f"  {f.stat().st_size/1024:6.0f} KB  figures/{f.name}")
""")

nb["cells"] = C
nb.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
nbf.write(nb, "analysis.ipynb")
print(f"wrote analysis.ipynb with {len(C)} cells")
