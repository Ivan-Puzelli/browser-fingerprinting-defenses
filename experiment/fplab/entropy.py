# SPDX-License-Identifier: MIT
"""Entropy budget per browser, and the anonymity-set capacity test.

Method (Eckersley 2010): take published per-attribute entropy from large public
datasets, then ask, per browser, whether the attribute is still USABLE by a
tracker. It pays out its published bits only if all three hold:

    present   - not null / empty / withheld
    stable    - same value across independent sessions (Level 3)
    genuine   - equals the real machine value, so it still separates THIS
                machine from other machines

Fail any one and it contributes 0 bits. What survives is the browser's residual
fingerprinting surface; the drop from the baseline is the entropy reduction.

Population entropy cannot be measured from one machine. Every weight below is a
*citation*, not a result. The dispositions beside them are what we measured.
"""
import collections

from .data import ARMS, BASELINE, FAMILY, attr_value, components, select, value

# Four independent weightings, so the same measurement can be priced four ways.
#   pano    Eckersley 2010, Panopticlick,        N = 470,161
#   amiu    Laperdrix et al. 2016, AmIUnique,    N = 118,934
#   hiding  Gomez-Boix et al. 2018,              N = 2,067,942
#   berke   Berke et al., PoPETs 2025(1), Tab.1, N = 8,400 (US)
# The first three are transcribed from Table 3 of Laperdrix et al.'s survey
# (ACM TWEB 2020), which collates all three studies on one page.
#   `frozen` records that a published weight no longer applies to a 2026
#           browser, with the reason; None means it still does.
#   `verify` marks a weight worth re-checking against its source before citing.
# Two mappings are approximations, flagged here rather than buried:
#   webGlBasics -> "WebGL Renderer", the dominant term of the FingerprintJS
#                  component ("WebGL Vendor" is listed separately, 2.14/2.28).
#                  For berke, the Unmasked Renderer row only - taking all four
#                  of Berke's WebGL rows would double-count one surface.
#   languages   -> "Content language"
BITS = {
    "userAgent": dict(
        pano=10.00, amiu=9.78, hiding=7.15, berke=4.613, verify=False, frozen=None,
        source="Laperdrix survey 2020, Table 3"),
    "plugins": dict(
        pano=15.40, amiu=11.06, hiding=9.49, berke=None, verify=False,
        source="Eckersley T1 / Laperdrix T3",
        frozen="navigator.plugins frozen to a spec-mandated PDF-viewer list "
               "(HTML spec, ~2020); one identical value on the 4 arms that do not "
               "farble it, so it discriminates nothing - Brave nonetheless still "
               "randomizes it, 40 distinct values in 40 sessions"),
    "fonts": dict(
        pano=13.90, amiu=8.38, hiding=6.90, berke=None, verify=False, frozen=None,
        source="Eckersley T1 / Laperdrix T3 (Flash)"),
    "screenResolution": dict(
        pano=4.83, amiu=4.89, hiding=4.85, berke=5.51, verify=False, frozen=None,
        source="Laperdrix survey 2020, Table 3"),
    "timezone": dict(
        pano=3.04, amiu=3.34, hiding=0.16, berke=2.064, verify=False, frozen=None,
        source="Laperdrix survey 2020, Table 3"),
    "supercookies": dict(
        pano=2.12, amiu=0.41, hiding=0.04, berke=None, verify=True,
        source="Eckersley T1 (storage probe)",
        frozen="localStorage/sessionStorage/indexedDB all enabled by default in "
               "every modern browser; true on all 6 arms"),
    "cookiesEnabled": dict(
        pano=0.353, amiu=0.25, hiding=0.00, berke=None, verify=False,
        source="Eckersley T1",
        frozen="true on all 6 arms; no longer discriminating"),
    "canvas": dict(
        pano=None, amiu=8.28, hiding=8.55, berke=None, verify=True, frozen=None,
        source="Laperdrix 2016 (absent from Eckersley)"),
    "webGlBasics": dict(
        pano=None, amiu=3.41, hiding=5.54, berke=6.833, verify=True, frozen=None,
        source="Laperdrix survey 2020, Table 3 (WebGL Renderer)"),
    "languages": dict(
        pano=None, amiu=5.92, hiding=2.72, berke=1.73, verify=True, frozen=None,
        source="Laperdrix 2016 (Content language)"),
    "platform": dict(
        pano=None, amiu=2.31, hiding=1.20, berke=2.114, verify=True, frozen=None,
        source="Laperdrix 2016"),
    "colorDepth": dict(
        pano=None, amiu=None, hiding=None, berke=0.616, verify=True, frozen=None,
        source="Berke et al., PoPETs 2025(1), Table 1"),
    "touchSupport": dict(
        pano=None, amiu=None, hiding=None, berke=1.463, verify=True,
        source="Berke et al., PoPETs 2025(1), Table 1",
        frozen="0 touch points on all 6 arms (desktop host)"),
    # Post-dates the older studies and appears in none of the three. Berke's
    # population differs from AmIUnique's, so carrying this one weight into the
    # amiu column is an approximation, and is flagged as such in the paper.
    "hardwareConcurrency": dict(
        pano=None, amiu=2.34, hiding=None, berke=2.34, verify=True, frozen=None,
        source="Berke et al., PoPETs 2025(1), Table 1 (N=8,400 US, FingerprintJS)"),
}

WEIGHTINGS = ("pano", "amiu", "hiding", "berke")

STUDY_N = {"pano": "470k", "amiu": "119k", "hiding": "2.07M", "berke": "8.4k"}

STUDY_NAME = {"pano": "Panopticlick 2010", "amiu": "AmIUnique",
              "hiding": "Hiding in the Crowd 2018", "berke": "Berke 2025 (US)"}

# A value differing from the local baseline is NOT evidence of a defense: it may
# be ordinary release skew. PINNED pays zero bits only where the substituted
# value is shared across the defense's user population. That cannot be shown
# from one machine, so it must be evidenced explicitly. Anything PINNED by the
# value test but absent from this table is scored LEAKED instead.
PIN_EVIDENCE = {
    # NOT "identical for all Tor users": a Windows Tor user emits
    # "Windows NT 10.0" where a mac user emits "Macintosh". Tor pins the
    # VERSION field uniformly across its population, so the OS class is the
    # string's only remaining variable - and that class is already charged to
    # `platform`, which is scored LEAKED for Tor. H(userAgent | platform) = 0,
    # so paying for it again would double-count one variable. The undefended
    # arms differ: their UA also carries the browser version, which varies
    # across the Firefox population, so their 9.78 bits stand. Approximate
    # during a release rollout, when Tor users straddle two versions.
    ("tor-standard", "userAgent"):
        "version pinned population-wide; residual OS class already paid by platform",
    # fonts is admitted under protest, exactly like screenResolution below:
    # Tor does not refuse enumeration, it FILTERS it. We measure
    # ['Helvetica Neue', 'Menlo'] against the host's four - a strict subset,
    # i.e. allowlist AND installed. A host without those faces reports fewer,
    # so the value is host-dependent and its residual entropy is > 0. Scoring
    # it 0 is an upper bound on the saving. Bracketed in the paper: withdrawing
    # this credit moves Tor from 90% to 73%.
    ("tor-standard", "fonts"):
        "ALLOWLIST-FILTERED - upper bound, see paper",
    ("tor-standard", "languages"):
        "Tor forces en-US for every user regardless of locale",
    ("tor-standard", "timezone"): 'constant: RFP and Tor both emit Atlantic/Reykjavik',
    ("firefox-rfp", "timezone"):  'constant: RFP and Tor both emit Atlantic/Reykjavik',
    ("tor-standard", "webGlBasics"): 'constant: RFP and Tor both emit "Mozilla"',
    ("firefox-rfp", "webGlBasics"):  'constant: RFP and Tor both emit "Mozilla"',
    # screenResolution is admitted under protest: letterboxing QUANTIZES the
    # real window onto a grid, it does not emit a constant. RFP reports
    # [700,1200] and Tor [600,1200] on the same host - a shared constant would
    # agree. Residual entropy is > 0 and not estimable from one machine, so
    # scoring it 0 is an upper bound on the saving. Bracketed in the paper.
    ("firefox-rfp", "screenResolution"): "QUANTIZED - upper bound, see paper",
    ("tor-standard", "screenResolution"): "QUANTIZED - upper bound, see paper",
    # brave/userAgent is absent on purpose: Brave emits a plain Chrome UA
    # differing from ours only in the Chromium major (150 vs 151). That is
    # release skew, not a defense, so it is scored LEAKED.
}

# Population of each defense's user base, for the capacity test. Chrome and
# Firefox are the undefended baselines, so the relevant crowd is their whole
# user base. Figures are order-of-magnitude and cited as such in the paper - but
# chrome and firefox must NOT share one, since their residual bits are identical
# and only the population separates them.
POPULATION = {
    "chrome": 3.5e9,                            # ~65% of ~5.4e9 web users
    "firefox": 1.8e8,                           # Mozilla published MAU
    "brave": 8.0e7, "brave-aggressive": 8.0e7,  # Brave reported MAU
    "tor-standard": 2.0e6,                      # Tor Metrics, daily users
    "firefox-rfp": None,                        # RFP users are Firefox users
}

# Upper bound on the correlation deflation, derived in the paper from
# Gomez-Boix's measured 33.6% desktop uniqueness at N = 2.07M.
DELTA_MAX = 2.33


def observe(rows, attr, arm, level=3):
    """-> (modal value, distinct values, observations) for one (attr, arm)."""
    obs = [o for o in (attr_value(r, attr) for r in select(rows, arm, level))
           if o is not None]
    if not obs:
        return None, 0, 0
    mode, _ = collections.Counter(obs).most_common(1)[0]
    return mode, len(set(obs)), len(obs)


def pin_evidenced(rows, attr, arm, mode, level=3):
    """Is the substituted value shared across the defense's user population?

    Two sources of evidence, and nothing else counts:

    1. Cross-product agreement. If a product from a DIFFERENT family emits the
       byte-identical substituted value, it is a hardcoded constant in shared
       code rather than a per-user draw. Firefox RFP and Tor Browser both emit
       Atlantic/Reykjavik, "Mozilla" and colorDepth 24; independent products
       cannot agree by chance.
    2. Documented uniformity, for defenses only one product implements - Tor's
       bundled fonts, its pinned user agent and its forced en-US.

    Differing from the local baseline is NOT evidence, which is why this test
    exists at all.
    """
    if (arm, attr) in PIN_EVIDENCE:
        return True
    return any(m is not None and m == mode
               for other in ARMS
               if FAMILY[other] != FAMILY[arm] and other != BASELINE[other]
               for m in (observe(rows, attr, other, level)[0],))


def classify(rows, attr, arm, level=3):
    """-> (state, distinct values). State is one of
    ABSENT / RANDOMIZED / WITHHELD / PINNED / LEAKED."""
    mode, k, _ = observe(rows, attr, arm, level)
    if mode is None:
        return "ABSENT", k
    if k > 1:
        return "RANDOMIZED", k
    if mode in ("null", '""', "[]", "{}", ""):
        return "WITHHELD", k
    base, _, _ = observe(rows, attr, BASELINE[arm], level)
    if base is not None and mode != base and pin_evidenced(rows, attr, arm, mode, level):
        return "PINNED", k
    return "LEAKED", k


def weighted_attrs(weighting, drop_frozen=True, drop=()):
    """Attributes carrying a weight under this pricing.

    `drop` removes further attributes, which is how the adaptive adversary of
    the paper is modelled: a tracker who knows a channel is noised simply stops
    reading it, and the budget has to be recomputed without it on BOTH sides.
    """
    return [a for a, s in BITS.items()
            if s[weighting] is not None
            and not (drop_frozen and s["frozen"])
            and a not in drop]


def budget(rows, weighting="amiu", drop_frozen=True, drop=()):
    """-> (table, totals). table[(arm, attr)] = (state, bits, gain, distinct);
    totals[arm] = (surviving bits, bits available)."""
    table, totals = {}, {}
    attrs = weighted_attrs(weighting, drop_frozen, drop)
    for arm in ARMS:
        surviving = available = 0.0
        for attr in attrs:
            bits = BITS[attr][weighting]
            state, distinct = classify(rows, attr, arm)
            available += bits
            # No observation is not evidence that a defense removed entropy.
            gain = float("nan") if state == "ABSENT" else bits if state == "LEAKED" else 0.0
            surviving += gain
            table[(arm, attr)] = (state, bits, gain, distinct)
        totals[arm] = (surviving, available)
    return table, totals


def reduction(totals):
    """Percent entropy removed, each arm against its own engine's baseline."""
    return {a: 100 * (1 - totals[a][0] / totals[BASELINE[a]][0])
            if totals[BASELINE[a]][0] else float("nan") for a in ARMS}


def classification_sensitivity(rows, weighting="amiu"):
    """Withdraw uncertain screen/font credits by charging their full weights.

    These are alternative scoring assumptions, not measured residual entropy.
    """
    import pandas as pd
    table, totals = budget(rows, weighting)
    screens = [(a, "screenResolution") for a in ("firefox-rfp", "tor-standard")]
    fonts = [("tor-standard", "fonts")]
    cases = {"Default credits": [], "Screen charged": screens,
             "Fonts charged": fonts, "Both charged": screens + fonts}
    out = {}
    for name, restored in cases.items():
        adjusted = dict(totals)
        for arm, attr in restored:
            if (arm, attr) in table:
                _, bits, gain, _ = table[(arm, attr)]
                surviving, available = adjusted[arm]
                adjusted[arm] = (surviving + bits - gain, available)
        out[name] = reduction(adjusted)
    return pd.DataFrame(out).T


# Composite budget attributes, excluded from the unweighted count because each
# is synthesised from components that the count already includes individually.
COMPOSITE = {"supercookies"}   # = localStorage + sessionStorage + indexedDB


def unweighted_attrs(rows):
    """The 43 attributes counted by the unweighted ruler: every collected
    component, plus `userAgent`, which is read from the request header and so is
    not a component. Composites are left out to avoid counting them twice."""
    return sorted((set(components(rows)) | set(BITS)) - COMPOSITE)


def unweighted_reduction(rows, level=3):
    """A fifth ruler that borrows nothing: count how many attributes still leak,
    treating every one alike, so `colorDepth` and `canvas` weigh the same. Uses
    the same strict classifier, so a pinned value must still be evidenced."""
    attrs = unweighted_attrs(rows)
    states = {arm: [classify(rows, a, arm, level)[0] for a in attrs] for arm in ARMS}
    leaked = {arm: float("nan") if "ABSENT" in ss else ss.count("LEAKED")
              for arm, ss in states.items()}
    return leaked, {a: 100 * (1 - leaked[a] / leaked[BASELINE[a]])
                    if leaked[BASELINE[a]] else float("nan") for a in ARMS}, len(attrs)


def anonymity_set(h, n, delta=1.0):
    """Modelled crowd size under assumed effective entropy, not observed uniqueness."""
    return None if n is None else n / 2.0 ** (h / delta)


def capacity(rows, weighting="amiu", drop_frozen=True, deltas=(1.0, 2.0), drop=()):
    """The capacity test: does the residual fit inside the browser's own crowd?

    Reduction percentages say how much entropy a defense removed; they cannot
    say whether it removed *enough*. Lying moves a user into the crowd of
    everyone telling the same lie, so the crowd must be larger than
    2**(residual bits) for those bits not to single them out inside it.
    """
    _, totals = budget(rows, weighting, drop_frozen, drop)
    return {arm: dict(bits=totals[arm][0], population=POPULATION[arm],
                      k={d: anonymity_set(totals[arm][0], POPULATION[arm], d)
                         for d in deltas})
            for arm in ARMS}


# The channels Brave farbles. An adaptive adversary discards them precisely
# because they are noisy, which costs it their bits but costs the defense all of
# its benefit - the subject of the paper's adaptive-adversary section.
FARBLED = ("canvas", "audio", "plugins", "hardwareConcurrency", "screenResolution")


def drop_channels(rows, weighting="amiu", drop=("canvas",)):
    """Residual bits and reduction once the tracker stops reading `drop`.

    Returns (bits per arm, reduction per arm, baseline bits), all recomputed
    with the dropped attributes removed from the budget on both sides.
    """
    _, totals = budget(rows, weighting, drop=drop)
    bits = {a: totals[a][0] for a in ARMS}
    return bits, reduction(totals), totals["chrome"][1]


def drop_channel_sweep(rows, weighting="amiu", channels=FARBLED):
    """Residual bits per arm as each *priced* farbled channel is discarded in
    turn, then all of them together.

    `audio` carries no published weight and `plugins` is frozen out of the
    budget, so discarding either moves no arm by construction. They get no
    scenario column: one that cannot move reads as evidence that the channel is
    worthless, which is a different claim from it never having been priced.
    """
    out = {"none": drop_channels(rows, weighting, drop=())[0]}
    for c in channels:
        if c in BITS and not BITS[c]["frozen"]:
            out[f"-{c}"] = drop_channels(rows, weighting, drop=(c,))[0]
    out["-all farbled"] = drop_channels(rows, weighting, drop=channels)[0]
    return out
