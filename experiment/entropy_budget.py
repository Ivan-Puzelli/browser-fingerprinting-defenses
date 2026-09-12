#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Entropy budget per browser - command-line report.

All logic lives in `fplab.entropy`; this file only prints it. See that module
for the method, the weightings and the evidence rules.

    python entropy_budget.py            # full report
    python entropy_budget.py --export   # also write figures/entropy_budget.png
"""
import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from fplab import data, entropy as E, figures

W = 94


def rule(title):
    print(f"\n{'=' * W}\n{title}\n{'=' * W}")


def report_budget(rows, weighting, drop_frozen):
    table, totals = E.budget(rows, weighting, drop_frozen)
    attrs = E.weighted_attrs(weighting, drop_frozen)
    red = E.reduction(totals)
    tag = ("corrected (frozen attributes dropped)" if drop_frozen
           else "naive (published weights as-is)")
    rule(f"{weighting.upper()} weighting, {E.STUDY_NAME[weighting]} "
         f"(N={E.STUDY_N[weighting]}) - {tag}")
    print(f"{'attribute':22}{'bits':>7}  " + "".join(f"{a[:11]:>12}" for a in E.ARMS))
    for attr in attrs:
        print(f"{attr:22}{E.BITS[attr][weighting]:7.2f}  " +
              "".join(f"{table[(arm, attr)][0][:11]:>12}" for arm in E.ARMS))
    print("-" * W)
    print(f"{'SURVIVING BITS':22}{totals['chrome'][1]:7.2f}  " +
          "".join(f"{totals[a][0]:>12.2f}" for a in E.ARMS))
    print(f"{'reduction vs baseline':22}{'':7}  " +
          "".join(f"{red[a]:>11.0f}%" for a in E.ARMS))
    return totals


def report_robustness(rows):
    """The same measurement priced five ways - the paper's robustness table."""
    rule("ROBUSTNESS - identical dispositions, five different price lists")
    print(f"{'weighting':26}{'N':>8}" + "".join(f"{a[:11]:>13}" for a in E.ARMS))
    for w in E.WEIGHTINGS:
        red = E.reduction(E.budget(rows, w)[1])
        print(f"{E.STUDY_NAME[w]:26}{E.STUDY_N[w]:>8}" +
              "".join(f"{red[a]:>12.0f}%" for a in E.ARMS))
    leaked, red, n = E.unweighted_reduction(rows)
    print(f"{'unweighted, ' + str(n) + ' attributes':26}{'---':>8}" +
          "".join(f"{red[a]:>12.0f}%" for a in E.ARMS))
    print(f"{'  (attributes leaking)':26}{'':>8}" +
          "".join(f"{leaked[a]:>13}" for a in E.ARMS))


def report_capacity(rows, weighting="amiu"):
    cap = E.capacity(rows, weighting)
    rule("ANONYMITY-SET CAPACITY - does the residual fit the crowd it hides in?")
    print("k = N / 2**(H/delta): modelled crowd size under assumed effective entropy.\n"
          "These are sensitivity scenarios, not measured anonymity sets.\n"
          "k < 1 does not by itself prove individual identification.\n")
    print(f"{'arm':20}{'H_res':>8}{'N':>12}{'k (d=1)':>13}{'k (d=2)':>13}   verdict")
    for arm in E.ARMS:
        c = cap[arm]
        if c["population"] is None:
            print(f"{arm:20}{c['bits']:>8.2f}{'---':>12}{'---':>13}{'---':>13}   "
                  "no separate population")
            continue
        k1, k2 = c["k"][1.0], c["k"][2.0]
        verdict = ("k < 1 at both" if k2 < 1 else
                   "k < 1 only at d=1" if k1 < 1 else "k >= 1 at both")
        print(f"{arm:20}{c['bits']:>8.2f}{c['population']:>12.1e}"
              f"{k1:>13.2e}{k2:>13.2e}   {verdict}")
    h = cap["firefox-rfp"]["bits"]
    print(f"\n  RFP's user count is unavailable. Modelled k >= 1 requires "
          f"{2**h:,.0f} users at delta=1, or {2**(h/2):,.0f} at delta=2.\n"
          "  Neither scenario establishes RFP's actual crowd size.")
    return cap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weighting", default="amiu", choices=E.WEIGHTINGS,
                    help="price list for the capacity test and the export (default: amiu)")
    ap.add_argument("--export", action="store_true",
                    help="also write figures/entropy_budget.png")
    args = ap.parse_args()

    rows = data.load()

    print("\nFROZEN attributes excluded from the corrected budget:")
    for a, s in E.BITS.items():
        if s["frozen"]:
            print(f"  {a:20} {s['pano'] or 0:5.2f} bits (Panopticlick)  - {s['frozen']}")

    for w in E.WEIGHTINGS:
        naive = report_budget(rows, w, drop_frozen=False)
        corr = report_budget(rows, w, drop_frozen=True)
        n, c = naive["chrome"][0], corr["chrome"][0]
        if n:
            print(f"\n  >> stale-weight inflation: chrome scores {n:.1f} bits naively "
                  f"vs {c:.1f} corrected ({100*(n-c)/n:.0f}% overstated)")

    report_robustness(rows)
    report_capacity(rows, args.weighting)

    rule("DISTINCT VALUES per (arm, attribute) at Level 3 - what each defense randomizes")
    for arm in E.ARMS:
        var = [(a, k) for a in E.BITS for st, k in (E.classify(rows, a, arm),) if k > 1]
        if var:
            print(f"  {arm:20} " + ", ".join(f"{a}={k}" for a, k in var))

    if args.export:
        print()
        table, totals = E.budget(rows, args.weighting)
        attrs = E.weighted_attrs(args.weighting)
        figures.entropy_budget(table, totals, attrs, args.weighting)


if __name__ == "__main__":
    main()
