# SPDX-License-Identifier: MIT
"""
Local test site for the cross-browser fingerprinting-defense experiment.

One Flask process serves two *virtual* domains, distinguished only by the Host
header: site-a.test and site-b.test (both resolve to 127.0.0.1 via /etc/hosts).
Two hostnames on one process is what makes Level 4 possible: Brave derives its
farbling domain_key from the eTLD+1, so site-a.test and site-b.test must be
genuinely different origins to the browser, while remaining the same server to
us.

No HTTPS, per the brief. Two consequences, both verified rather than assumed:

  * `http://site-a.test` is **not** a secure context. The secure-context rule
    keys on the *host string* (`localhost`, `127.0.0.1`, `[::1]`), not on the
    address it resolves to, so pointing site-a.test at 127.0.0.1 does not buy
    it trustworthiness. Measured in Chrome 151: `window.isSecureContext` is
    `false` and `crypto.subtle` is `undefined` on site-a.test, `true`/`object`
    on 127.0.0.1. Nothing under test is gated on this — FingerprintJS 5.2.0
    uses no secure-context-only API and never touches `crypto.subtle` (its
    hashing is pure JS) — so the surfaces measured here are unaffected. It is
    recorded because it is the kind of assumption that quietly invalidates a
    result if left unchecked.
  * Tor Browser at security level *Safer* withholds the `script` capability
    from non-HTTPS origins, so on this server Safer would block JavaScript
    outright. See FINDINGS.md; this is why Safer cannot be measured here even
    by hand.

Usage:  python server.py [--port 8000] [--results data/results.jsonl]
"""

import argparse
import json
import os
import threading
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, request, send_from_directory
from markupsafe import escape

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

app = Flask(__name__, static_folder=None)

# Appending from several concurrent WebDriver sessions, so serialise writes.
_write_lock = threading.Lock()
RESULTS_PATH = os.path.join(HERE, "data", "results.jsonl")


@app.after_request
def _no_cache(response: Response) -> Response:
    """
    Caching would silently defeat Level 2. A reload that serves probe.html or
    fp.umd.js from cache is not an independent execution of the probe, and a
    cached bundle can even keep JS state alive across what should be separate
    page loads. Every response is therefore explicitly uncacheable.
    """
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
@app.route("/probe.html")
def probe():
    """
    Serves the probe page, injecting this run's metadata into the <noscript>
    beacon URL. The beacon cannot build its own query string — that would need
    the JavaScript that is switched off in precisely the case it exists for.
    """
    with open(os.path.join(STATIC_DIR, "probe.html"), encoding="utf-8") as fh:
        html = fh.read()
    html = html.replace("__QS__", escape(request.query_string.decode("utf-8", "replace")))
    return Response(html, mimetype="text/html")


@app.route("/canvas_utility.html")
def canvas_utility():
    """
    Standalone usability demonstration (capture_canvas_utility.py). It records
    nothing and shares no code with the probe, so it cannot perturb the
    measured dataset.
    """
    return send_from_directory(STATIC_DIR, "canvas_utility.html", mimetype="text/html")


@app.route("/fp.umd.js")
def bundle():
    return send_from_directory(STATIC_DIR, "fp.umd.js", mimetype="application/javascript")


@app.route("/health")
def health():
    return jsonify(status="ok", results=RESULTS_PATH)


@app.route("/collect-passive")
def collect_passive():
    """
    Fired by the <noscript> beacon in probe.html. Records the fingerprint that
    survives with JavaScript switched off — which is the only channel left on
    Tor Browser at security level "Safest". Deliberately a GET of an image, so
    it works with no scripting whatsoever.
    """
    def as_int(name, default):
        # The beacon URL is built by string substitution in probe.html, so a
        # malformed run/level must degrade to a recorded row, not a 500 that
        # loses the only observation a no-JS browser can give us.
        try:
            return int(request.args.get(name, default) or default)
        except (TypeError, ValueError):
            return default

    payload = {
        "browser": request.args.get("browser", "unknown-nojs"),
        "level": as_int("level", 1),
        "run_index": as_int("run", 0),
        "automated": request.args.get("automated") == "1",
        "domain": (request.headers.get("Host", "") or "").split(":")[0],
        "js_executed": False,
        "components": {},
        "visitorId": None,
        "note": "JavaScript disabled; passive header-only observation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "_server_ts": datetime.now(timezone.utc).isoformat(),
        "_host_header": request.headers.get("Host", ""),
        "_remote_addr": request.remote_addr or "",
        "_user_agent_header": request.headers.get("User-Agent", ""),
        "_accept_header": request.headers.get("Accept", ""),
        "_accept_language_header": request.headers.get("Accept-Language", ""),
        "_accept_encoding_header": request.headers.get("Accept-Encoding", ""),
        "_sec_ch_ua_header": request.headers.get("Sec-CH-UA", ""),
        "_header_order": list(request.headers.keys()),
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with _write_lock:
        with open(RESULTS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    # 1x1 transparent GIF
    gif = (b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!"
           b"\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01"
           b"\x00\x00\x02\x02D\x01\x00;")
    return Response(gif, mimetype="image/gif")


@app.route("/collect", methods=["POST"])
def collect():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify(error="invalid json"), 400

    # Server-observed metadata. Recorded server-side on purpose: these are the
    # signals a real tracker sees *without* running any JavaScript, which is the
    # only channel still open on Tor at security level "Safest" (see FINDINGS.md).
    payload["_server_ts"] = datetime.now(timezone.utc).isoformat()
    payload["_host_header"] = request.headers.get("Host", "")
    payload["_remote_addr"] = request.remote_addr or ""
    payload["_user_agent_header"] = request.headers.get("User-Agent", "")
    payload["_accept_header"] = request.headers.get("Accept", "")
    payload["_accept_language_header"] = request.headers.get("Accept-Language", "")
    payload["_accept_encoding_header"] = request.headers.get("Accept-Encoding", "")
    payload["_sec_ch_ua_header"] = request.headers.get("Sec-CH-UA", "")
    payload["_header_order"] = list(request.headers.keys())

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with _write_lock:
        with open(RESULTS_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return jsonify(ok=True)


def main():
    global RESULTS_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--results", default=RESULTS_PATH)
    args = parser.parse_args()
    RESULTS_PATH = os.path.abspath(args.results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    # threaded=True: Level 4 visits two hosts in quick succession, and the
    # Selenium driver can have a request in flight while /collect is writing.
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
