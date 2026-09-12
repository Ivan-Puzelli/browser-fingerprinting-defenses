# SPDX-License-Identifier: MIT
"""
Selenium collection for the 4-level fingerprinting-stability design.

The four levels are nested repetitions of the *same* probe, differing only in
what is held constant between measurements:

  Level 1  same script execution, two consecutive canvas calls
           -> embedded in every single probe run; no separate loop.
           Tests raw within-execution determinism.
  Level 2  same browser process, N reloads, no restart
           -> one WebDriver session per (browser, domain), N page loads.
           Tests "deterministic within a session" (notes property 3).
  Level 3  fresh browser process per run
           -> N WebDriver sessions per (browser, domain), 1 page load each,
              each with its own throwaway profile directory.
           Tests "session-specific" (notes property 2).
  Level 4  same process, both domains
           -> one session per browser, alternating site-a.test / site-b.test.
           Tests "site-specific" (notes property 1).

Usage:
    python run_experiment.py --browsers chrome brave firefox firefox-rfp
    python run_experiment.py --n 2 --smoke      # quick pipeline check
"""

import argparse
import contextlib
import os
import shutil
import sys
import tempfile
import time
import traceback

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait

HERE = os.path.dirname(os.path.abspath(__file__))
BRAVE_BINARY = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
TOR_BINARY = "/Applications/Tor Browser.app/Contents/MacOS/firefox"

# Tor Browser security levels, as stored in the security-slider pref.
# 4 = Standard, 2 = Safer, 1 = Safest. Only Standard is analysed. Safer is out
# of scope (its slider is enforced by a bundled extension that never initialises
# in a WebDriver profile, so the pref is inert - see FINDINGS.md), and the
# no-JavaScript arm is retained only so the collected dataset stays reproducible:
# it yields headers and no components, and the analysis drops it.
TOR_SECURITY_LEVELS = {"tor-standard": 4, "tor-nojs": 1}

DOMAINS = ["site-a.test", "site-b.test"]
PAGE_TIMEOUT = 45


# ── Brave Shields ────────────────────────────────────────────────────────────
# Brave stores Shields state as Chromium *content settings* in the profile's
# Default/Preferences JSON, under profile.content_settings.exceptions. Writing
# that file before first launch is the scriptable equivalent of the Shields UI.
#
# The fingerprinting level is encoded with Brave's "fake pattern" convention:
# the primary pattern is a sentinel URL, not a real site.
#   fingerprintingV2 "*" = BLOCK(2)         -> Strict / aggressive
#   fingerprintingV2 "https://balanced/*"   -> Standard (default farbling)
#   fingerprintingV2 "*" = ALLOW(1)         -> Disabled (no farbling)
#
# CAUTION, verified the hard way: an explicit `"*,*": 1` (ALLOW) means "allow
# fingerprinting", i.e. farbling OFF. Writing it alongside the balanced sentinel
# silently disables the very defense being measured — Brave then returns canvas,
# WebGL and audio values byte-identical to Chrome's. The "standard" preset must
# therefore carry the sentinel *only*. Confirmed by A/B test against an
# unseeded (pure Brave default) profile.
BRAVE_SHIELDS_PRESETS = {
    "standard": {"https://balanced/*,*": 2},
    "aggressive": {"*,*": 2},
    "off": {"*,*": 1},
}


def write_brave_shields(profile_dir: str, preset: str) -> None:
    """Pre-seed Shields fingerprinting settings into a fresh Brave profile."""
    import json
    from datetime import datetime, timezone

    default_dir = os.path.join(profile_dir, "Default")
    os.makedirs(default_dir, exist_ok=True)
    prefs_path = os.path.join(default_dir, "Preferences")

    prefs = {}
    if os.path.exists(prefs_path):
        try:
            with open(prefs_path, encoding="utf-8") as fh:
                prefs = json.load(fh)
        except (ValueError, OSError):
            prefs = {}

    # Chromium stores content-setting timestamps as microseconds since 1601.
    epoch_1601 = int((datetime.now(timezone.utc).timestamp() + 11644473600) * 1_000_000)

    exceptions = prefs.setdefault("profile", {}).setdefault("content_settings", {}).setdefault("exceptions", {})
    fp = exceptions.setdefault("fingerprintingV2", {})
    for pattern, setting in BRAVE_SHIELDS_PRESETS[preset].items():
        fp[pattern] = {"expiration": "0", "last_modified": str(epoch_1601), "model": 0, "setting": setting}

    with open(prefs_path, "w", encoding="utf-8") as fh:
        json.dump(prefs, fh)


# ── Driver factories ─────────────────────────────────────────────────────────

def _chromium_options(profile_dir: str, headless: bool, binary: str = "") -> ChromeOptions:
    opts = ChromeOptions()
    if binary:
        opts.binary_location = binary
    # A fresh --user-data-dir per Level-3 run is the whole point of Level 3:
    # reusing a profile would carry Brave's session_key, cookies and cache
    # across runs that are supposed to be independent samples.
    opts.add_argument(f"--user-data-dir={profile_dir}")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument("--disable-background-networking")
    opts.add_argument("--disable-sync")
    opts.add_argument("--window-size=1280,800")
    if headless:
        opts.add_argument("--headless=new")
    if binary == BRAVE_BINARY:
        # Brave 150 terminates at startup when ChromeDriver injects its default
        # `--test-type=webdriver` switch ("Mach rendezvous failed, terminating
        # process"). Bisected against the exact ChromeDriver argv; every other
        # default switch is harmless. Chrome is unaffected, so the exclusion is
        # applied only to Brave to keep the two arms otherwise identical.
        opts.add_experimental_option("excludeSwitches", ["test-type"])
    return opts


def _firefox_options(headless: bool, resist_fingerprinting: bool) -> FirefoxOptions:
    opts = FirefoxOptions()
    if headless:
        opts.add_argument("-headless")
    opts.set_preference("browser.shell.checkDefaultBrowser", False)
    opts.set_preference("datareporting.healthreport.uploadEnabled", False)
    opts.set_preference("toolkit.telemetry.enabled", False)
    if resist_fingerprinting:
        # The "strict" arm. Directly scriptable — no UI interaction needed.
        # RFP is Firefox's *uniformity* lever (spoofed UA, quantised window,
        # fixed timezone/locale) plus canvas/audio randomisation.
        opts.set_preference("privacy.resistFingerprinting", True)
        opts.set_preference("privacy.trackingprotection.enabled", True)
        opts.set_preference("browser.contentblocking.category", "strict")
        # RFP on a non-English locale pops a modal ("request English versions of
        # pages?") that blocks WebDriver on every single run. 2 = apply the
        # English spoof silently, which is the behaviour RFP is asking for
        # anyway; 1 would keep prompting.
        opts.set_preference("privacy.spoof_english", 2)
    return opts


def _tor_options(security_slider: int) -> FirefoxOptions:
    """
    Tor Browser is Firefox ESR with Marionette intact, so geckodriver drives it
    directly. Four things have to be neutralised first, none of which touch the
    fingerprinting surfaces under test — they only make a loopback server
    reachable from a browser built to refuse exactly that:

      1. the SOCKS proxy, which would send site-a.test into a Tor circuit that
         is not even bootstrapped (`DisableNetwork is set`);
      2. `network.dns.disabled`, which Tor sets so no DNS query can leak outside
         the circuit — with the proxy bypassed we need the OS resolver, which is
         what reads /etc/hosts;
      3. HTTPS-Only mode. Tor Browser runs in *permanent private-browsing mode*,
         so it is the `_pbm` variants of these prefs that actually govern it;
         the non-PBM ones alone have no effect;
      4. DoH (`network.trr.mode = 5`), for the same reason as (2).

    The security level itself is set through the slider pref, which is the
    scriptable equivalent of the Security Level UI panel.
    """
    o = FirefoxOptions()
    o.binary_location = TOR_BINARY
    o.set_preference("network.proxy.allow_hijacking_localhost", False)
    o.set_preference("network.proxy.no_proxies_on", "site-a.test,site-b.test,127.0.0.1,localhost")
    o.set_preference("network.dns.disabled", False)
    o.set_preference("network.trr.mode", 5)
    o.set_preference("dom.security.https_only_mode", False)
    o.set_preference("dom.security.https_first", False)
    o.set_preference("dom.security.https_only_mode_pbm", False)
    o.set_preference("dom.security.https_first_pbm", False)
    o.set_preference("browser.security_level.security_slider", security_slider)
    o.set_preference("extensions.torbutton.security_slider", security_slider)
    if security_slider == 1:  # Safest also disables JS outright
        o.set_preference("javascript.enabled", False)
    return o


def make_driver(browser: str, headless: bool):
    """Returns (driver, profile_dir_to_clean_up_or_None)."""
    if browser in ("chrome", "brave", "brave-aggressive"):
        profile_dir = tempfile.mkdtemp(prefix=f"fpexp-{browser}-")
        binary = BRAVE_BINARY if browser.startswith("brave") else ""
        if browser == "brave-aggressive":
            write_brave_shields(profile_dir, "aggressive")
        elif browser == "brave":
            write_brave_shields(profile_dir, "standard")
        opts = _chromium_options(profile_dir, headless, binary)
        driver = webdriver.Chrome(options=opts)
        return driver, profile_dir

    if browser in ("firefox", "firefox-rfp"):
        # Selenium 4 creates a fresh throwaway Firefox profile per session by
        # default, so Level-3 independence holds without extra bookkeeping.
        opts = _firefox_options(headless, resist_fingerprinting=(browser == "firefox-rfp"))
        driver = webdriver.Firefox(options=opts)
        return driver, None

    if browser in TOR_SECURITY_LEVELS:
        driver = webdriver.Firefox(options=_tor_options(TOR_SECURITY_LEVELS[browser]))
        return driver, None

    raise ValueError(f"unknown browser: {browser}")


# ── Probe driving ────────────────────────────────────────────────────────────

def probe_url(port: int, domain: str, browser: str, level: int, run_index: int, note: str = "") -> str:
    from urllib.parse import urlencode
    qs = urlencode({"browser": browser, "level": level, "run": run_index, "automated": "1", "note": note})
    return f"http://{domain}:{port}/probe.html?{qs}"


# Arms where JavaScript never runs, so there is no DOM state to wait for. The
# page's <noscript> beacon does the recording server-side instead.
NO_JS_BROWSERS = {"tor-nojs"}


def run_probe(driver, url: str, no_js: bool = False) -> str:
    """Loads the probe and waits for it to finish. Returns 'done', 'error' or 'no-js'."""
    driver.get(url)
    if no_js:
        # Nothing will ever set data-probe-state; just let the <noscript> image
        # beacon reach the server, then move on.
        time.sleep(2.5)
        return "no-js"
    WebDriverWait(driver, PAGE_TIMEOUT).until(
        lambda d: d.find_element("tag name", "body").get_attribute("data-probe-state") in ("done", "error")
    )
    state = driver.find_element("tag name", "body").get_attribute("data-probe-state")
    # The page sets done only after /collect acknowledges the saved observation.
    return state


def make_driver_resilient(browser: str, headless: bool, log, tries: int = 3):
    """
    Tor Browser bootstraps a tor daemon on every launch and occasionally exceeds
    geckodriver's startup window. A single slow launch must not abort a whole
    level, so retry a few times and let the caller skip the run if it still fails.
    """
    for attempt in range(tries):
        try:
            return make_driver(browser, headless)
        except Exception as exc:
            log(f"    driver attempt {attempt + 1}/{tries} failed: {str(exc).splitlines()[0][:110]}")
            time.sleep(4)
    return None, None


def cleanup(driver, profile_dir):
    try:
        driver.quit()
    except Exception:
        pass
    if profile_dir:
        shutil.rmtree(profile_dir, ignore_errors=True)


# ── The four levels ──────────────────────────────────────────────────────────
#
# Level 1 needs no runner: the probe draws the canvas twice inside a single
# execution, so every run of every level carries an L1 observation already.
#
# The three that do need a runner differ only in how sessions and domains are
# nested, so the session lifecycle and the per-run error handling live in the
# two helpers below and each level is left to state its own nesting.

@contextlib.contextmanager
def session(browser: str, headless: bool, log, what: str):
    """One browser session, always torn down and always cleaned up.

    Yields None if the browser would not start, so a single failed launch skips
    its runs instead of aborting the level.
    """
    driver, profile_dir = make_driver_resilient(browser, headless, log)
    if driver is None:
        log(f"    !! could not start {browser}; skipping {what}")
        yield None
        return
    try:
        yield driver
    finally:
        cleanup(driver, profile_dir)


def attempt(driver, url: str, no_js: bool, log, label: str) -> None:
    """One probe load. Reports failures and never raises: a lost run is a gap in
    the data, not a reason to lose the runs after it."""
    try:
        if run_probe(driver, url, no_js) == "error":
            log(f"    ! {label} reported a probe error")
    except Exception as exc:
        log(f"    ! {label} failed: {str(exc).splitlines()[0][:100]}")


def level2(browser: str, port: int, n: int, headless: bool, log) -> None:
    """Same process, N reloads, no restart. One session per (browser, domain).

    Tests determinism within a live session - farbling property P3."""
    no_js = browser in NO_JS_BROWSERS
    for domain in DOMAINS:
        log(f"  L2 {browser} {domain}: 1 session x {n} reloads")
        with session(browser, headless, log, f"L2 {domain}") as driver:
            if driver is None:
                continue
            for i in range(n):
                attempt(driver, probe_url(port, domain, browser, 2, i), no_js,
                        log, f"run {i}")


def level3(browser: str, port: int, n: int, headless: bool, log) -> None:
    """Fresh process and fresh profile per run. N sessions per (browser, domain).

    Tests session-specificity - farbling property P2."""
    no_js = browser in NO_JS_BROWSERS
    for domain in DOMAINS:
        log(f"  L3 {browser} {domain}: {n} sessions x 1 load")
        for i in range(n):
            with session(browser, headless, log, f"L3 run {i}") as driver:
                if driver is None:
                    continue
                attempt(driver, probe_url(port, domain, browser, 3, i), no_js,
                        log, f"run {i}")


def level4(browser: str, port: int, n: int, headless: bool, log) -> None:
    """Same process, both domains, alternating.

    The session is held constant so that the only thing varying is the eTLD+1 -
    which is exactly what Brave's domain_key keys on. Tests site-specificity,
    farbling property P1."""
    no_js = browser in NO_JS_BROWSERS
    log(f"  L4 {browser}: 1 session x {n} site-a/site-b pairs")
    with session(browser, headless, log, "L4") as driver:
        if driver is None:
            return
        for i in range(n):
            for domain in DOMAINS:
                attempt(driver, probe_url(port, domain, browser, 4, i, note="cross-domain-pair"),
                        no_js, log, f"pair {i} {domain}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browsers", nargs="+",
                        default=["chrome", "brave", "brave-aggressive", "firefox",
                                 "firefox-rfp", "tor-standard"])
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--n", type=int, default=20, help="repetitions per (browser, domain) for L2/L3")
    parser.add_argument("--n4", type=int, default=5, help="cross-domain pairs for L4")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    args = parser.parse_args()

    if args.smoke:
        args.n, args.n4 = 2, 1

    def log(msg):
        print(msg, flush=True)

    runners = {2: level2, 3: level3, 4: lambda b, p, n, h, l: level4(b, p, args.n4, h, l)}

    unknown_levels = [lv for lv in args.levels if lv not in runners]
    if unknown_levels:
        parser.error(f"--levels: no runner for {unknown_levels}; choose from {sorted(runners)} "
                     "(Level 1 is measured inside every probe execution, not as a separate run)")
    unknown_browsers = [b for b in args.browsers
                        if b not in ("chrome", "brave", "brave-aggressive", "firefox", "firefox-rfp")
                        and b not in TOR_SECURITY_LEVELS]
    if unknown_browsers:
        parser.error(f"--browsers: unknown {unknown_browsers}. 'tor-safer' and 'safari' "
                     "are out of scope (see FINDINGS.md).")

    started = time.time()
    for browser in args.browsers:
        log(f"[{browser}]")
        for level in args.levels:
            try:
                runners[level](browser, args.port, args.n, args.headless, log)
            except Exception:
                log(f"  !! {browser} level {level} aborted:")
                traceback.print_exc()
    log(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
