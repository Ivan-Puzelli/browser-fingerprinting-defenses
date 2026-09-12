#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Checks whether the QR exported by each browser still carries its payload.

Rather than trusting a decoder, this reads the module grid straight out of the
exported PNG and compares it to the reference matrix that segno produced for
the same payload. A grid that matches the reference exactly IS the code: any
conforming reader decodes it. A grid that does not match is not a degraded
code, it is a different image.

    python capture_canvas_utility.py     # writes figures/qr_export_*.png
    python verify_qr_export.py
"""

import os
import sys

import segno
from PIL import Image

# Must match drawQR() in static/canvas_utility.html.
PAYLOAD = "Export intact - Data Protection"
ECC = "q"
CANVAS = (330, 190)
PAD, BOX = 16, 150

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
EXPORTS = [("brave", "qr_export_brave.png"), ("tor-standard", "qr_export_tor-standard.png")]


def reference_matrix():
    q = segno.make(PAYLOAD, error=ECC)
    return [[1 if c else 0 for c in row] for row in q.matrix]


def read_matrix(path: str, n: int):
    """Samples the exported PNG at the centre of every module."""
    img = Image.open(path).convert("L")
    # The export is the canvas backing store, so it may be scaled but never
    # cropped; work in fractions of the canvas the page declared.
    sx, sy = img.width / CANVAS[0], img.height / CANVAS[1]
    cell = BOX / n
    x0, y0 = PAD, (CANVAS[1] - BOX) / 2
    out = []
    for r in range(n):
        row = []
        for c in range(n):
            px = int((x0 + (c + 0.5) * cell) * sx)
            py = int((y0 + (r + 0.5) * cell) * sy)
            row.append(1 if img.getpixel((px, py)) < 128 else 0)
        out.append(row)
    return out


def main() -> int:
    ref = reference_matrix()
    n = len(ref)
    total = n * n
    print(f"payload   : {PAYLOAD!r}")
    print(f"reference : version-{segno.make(PAYLOAD, error=ECC).version} QR, "
          f"ECC {ECC.upper()}, {n}x{n} = {total} modules\n")

    failed = False
    for arm, filename in EXPORTS:
        path = os.path.join(FIGURES, filename)
        if not os.path.exists(path):
            print(f"{arm:14} missing {filename} - run capture_canvas_utility.py first")
            failed = True
            continue
        got = read_matrix(path, n)
        same = sum(got[r][c] == ref[r][c] for r in range(n) for c in range(n))
        verdict = "carries the payload" if same == total else "does not encode the payload"
        print(f"{arm:14} {same}/{total} modules match  ({100 * same / total:.1f}%)  -> {verdict}")
        if arm == "brave" and same != total:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
