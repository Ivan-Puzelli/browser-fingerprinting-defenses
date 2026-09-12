#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pixel-level comparison of canvas defenses - command-line report.

All logic lives in `fplab.canvas`; this file only prints it and draws the
figure. Separates a sparse +/-1 mask applied at the API boundary (Brave) from
large readout changes (Gecko RFP and Tor) - a distinction
stability rates cannot make.

    python canvas_mechanism.py            # report
    python canvas_mechanism.py --no-plot  # report only
"""
import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from fplab import canvas, data, figures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=int, default=3,
                    help="repetition level to compare across (default: 3, independent sessions)")
    ap.add_argument("--no-plot", action="store_true", help="skip figures/canvas_mechanism.png")
    args = ap.parse_args()

    rows = data.load()
    m = canvas.mechanism(rows, args.level)

    print(f"\n{'arm':18}{'% px changed':>14}{'max delta':>11}{'only +/-1':>11}{'pairs':>7}"
          f"   mechanism")
    for arm, d in m.items():
        kind = ("no change" if d["frac_changed"] == 0
                else "sparse +/-1 mask" if d["only_pm1"] else "large readout changes")
        print(f"{arm:18}{d['pct_changed']:>13.2f}%{d['max_delta']:>11}"
              f"{str(d['only_pm1']):>11}{d['n_pairs']:>7}   {kind}")

    print("\nBrave's three farbling properties, measured on pixels rather than hashes:")
    for arm, p in canvas.properties(rows).items():
        print(f"  {arm}")
        print(f"    P3 within session, same site : {100*p['p3_within_session']:.2f}% changed")
        print(f"    P1 cross-domain, same session: {100*p['p1_cross_domain']:.2f}% changed "
              f"(max delta {p['p1_max_delta']})")
        print(f"    P2 across sessions           : {100*p['p2_cross_session']:.2f}% changed "
              f"(max delta {p['p2_max_delta']})")

    if not args.no_plot:
        print()
        figures.canvas_mechanism(canvas.mechanism_frame(rows, args.level))


if __name__ == "__main__":
    main()
