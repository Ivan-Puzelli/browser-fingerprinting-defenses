#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Screenshots static/canvas_utility.html in Brave and Tor Browser.

Produces exactly two files, each self-contained: the page shows the same
canvas as the compositor paints it and as toDataURL() returns it, so a
single screenshot carries its own control and no stitching is needed.

Reuses make_driver()/cleanup() from run_experiment.py rather than rebuilding
the driver stack, so Brave's shields preset and Tor's proxy/HTTPS-only
neutralisation stay defined in one place. Writes nothing to results.jsonl.

    python server.py                      # in another shell
    python capture_canvas_utility.py

The measured numbers are also printed, so the figure and the text can cite
the same run.
"""

import argparse
import base64
import os
import sys
import time

from selenium.webdriver.support.ui import WebDriverWait

from run_experiment import cleanup, make_driver

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")

# Wide enough that the two panes stay side by side after Tor letterboxes the
# viewport down to the next grid step. Height is deliberately larger than the
# host screen: the window manager clamps it, and full_page_screenshot() below
# captures past the fold anyway.
WINDOW = (1180, 1200)


def full_page_screenshot(driver, path: str) -> str:
    """
    The demo page is taller than any viewport this host can show - the screen
    is 832px and Tor letterboxes what is left - so a plain save_screenshot()
    cuts the measurements card off the bottom. Both engines can capture past
    the fold, by different routes.
    """
    if hasattr(driver, "get_full_page_screenshot_as_file"):   # Gecko
        try:
            if driver.get_full_page_screenshot_as_file(path):
                return "full page, gecko"
        except Exception:
            pass
    try:                                                       # Chromium, CDP
        metrics = driver.execute_cdp_cmd("Page.getLayoutMetrics", {})
        box = metrics.get("cssContentSize") or metrics["contentSize"]
        shot = driver.execute_cdp_cmd("Page.captureScreenshot", {
            "format": "png",
            "captureBeyondViewport": True,
            "clip": {"x": 0, "y": 0, "width": box["width"],
                     "height": box["height"], "scale": 1},
        })
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(shot["data"]))
        return "full page, cdp"
    except Exception:
        pass
    if not driver.save_screenshot(path):                       # last resort
        raise RuntimeError("save_screenshot returned false")
    return "VIEWPORT ONLY - page is cut off"

ARMS = [
    ("brave", "canvas_utility_brave.png"),
    ("tor-standard", "canvas_utility_tor.png"),
]


# The page is a centred 720px column; the two browsers give it different
# viewport widths (Tor letterboxes) and different pixel ratios, so raw
# screenshots do not sit side by side honestly. Crop each to that column and
# render both at one width, changing framing only - never content.
CONTENT_CSS_WIDTH = 756          # .wrap max-width plus body padding
PRESENTATION_WIDTH = 1500


def normalise(path: str, css_viewport_width: int) -> tuple:
    from PIL import Image
    img = Image.open(path)
    ratio = img.width / css_viewport_width
    band = min(int(CONTENT_CSS_WIDTH * ratio), img.width)
    left = (img.width - band) // 2
    img = img.crop((left, 0, left + band, img.height))
    height = round(img.height * PRESENTATION_WIDTH / img.width)
    img = img.resize((PRESENTATION_WIDTH, height), Image.LANCZOS)
    img.save(path)
    return img.size


def capture(browser: str, filename: str, port: int, domain: str) -> dict:
    url = f"http://{domain}:{port}/canvas_utility.html"
    print(f"[{browser}] {url}")
    driver, profile_dir = make_driver(browser, headless=False)
    try:
        driver.set_window_size(*WINDOW)
        driver.get(url)
        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script("return window.__demoReady === true")
        )
        # The <img> panes decode asynchronously from their data: URI; the ready
        # flag only says the script finished, not that the browser has painted.
        WebDriverWait(driver, 15).until(
            lambda d: d.execute_script(
                "return Array.from(document.images).every(i => i.complete && i.naturalWidth > 0)"
            )
        )
        time.sleep(1.0)

        results = driver.execute_script("return window.__demoResults")

        # The exported QR, saved as the file a real app would upload. Kept so
        # the claim "this one still scans" can be checked offline instead of
        # asserted (verify_qr_export.py).
        qr_uri = driver.execute_script(
            "return document.getElementById('watermark-preview').src")
        qr_path = os.path.join(FIGURES, f"qr_export_{browser}.png")
        with open(qr_path, "wb") as fh:
            fh.write(base64.b64decode(qr_uri.split(",", 1)[1]))
        results["qrExport"] = qr_path
        path = os.path.join(FIGURES, filename)
        how = full_page_screenshot(driver, path)
        css_width = driver.execute_script("return document.documentElement.clientWidth")
        size = normalise(path, css_width)

        print(f"  toDataURL() x2 identical : {results['identicalExports']}")
        print(f"  pixels differing         : {results['pixelsDiffering']:,}"
              f" / {results['pixelsTotal']:,}  ({results['pixelsPct']}%)")
        print(f"  max channel delta        : {results['maxChannelDelta']}")
        print(f"  exported PNG             : {results['pngBytes']:,} bytes")
        print(f"  -> {os.path.relpath(path, HERE)}  ({how}, {size[0]}x{size[1]})")
        results["browser"] = browser
        results["screenshot"] = path
        return results
    finally:
        cleanup(driver, profile_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--domain", default="site-a.test")
    ap.add_argument("--browsers", nargs="+", default=[a for a, _ in ARMS],
                    help="subset of: " + ", ".join(a for a, _ in ARMS))
    args = ap.parse_args()

    os.makedirs(FIGURES, exist_ok=True)
    wanted = dict(ARMS)

    collected, failed = [], []
    for browser in args.browsers:
        if browser not in wanted:
            print(f"!! unknown arm {browser}; choose from {sorted(wanted)}")
            failed.append(browser)
            continue
        try:
            collected.append(capture(browser, wanted[browser], args.port, args.domain))
        except Exception as exc:
            print(f"!! {browser} failed: {type(exc).__name__}: {exc}")
            failed.append(browser)

    if len(collected) == 2:
        a, b = collected
        print("\nsummary")
        for r in (a, b):
            verdict = "readout reproducible" if r["identicalExports"] else "readout differs per call"
            print(f"  {r['browser']:14} {verdict:26} "
                  f"{r['pixelsPct']}% of pixels differ between two reads, "
                  f"max delta {r['maxChannelDelta']}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
