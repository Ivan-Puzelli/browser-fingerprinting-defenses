# Browser Fingerprinting Lab

A local laboratory for collecting browser fingerprints and comparing their
behavior across repeated reads, page reloads, fresh browser sessions, and two
test domains. Browser automation uses Selenium; a Flask server serves the probe
and records observations as JSON Lines.

## Code layout

The application code lives in `experiment/`:

| File or directory | Responsibility |
|---|---|
| `server.py` | Serve the probe and bundle; append observations and HTTP request metadata to a dataset. |
| `run_experiment.py` | Configure browsers and run the collection loops. |
| `static/probe.html` | Collect raw canvas reads and FingerprintJS components, then send them to the server. |
| `static/fp.umd.js` | Prebuilt FingerprintJS 5.2.0 bundle used by the probe. |
| `patches/` | Instrumentation changes to the upstream FingerprintJS source. |
| `fplab/` | Shared loading, normalization, stability, pixel-comparison, entropy-budget, and similarity functions. |
| `entropy_budget.py` | Command-line interface to the weighted analysis. |
| `canvas_mechanism.py` | Command-line interface to pixel comparisons. |
| `build_analysis_nb.py` | Generate the analysis notebook from its source cells. |
| `analysis.ipynb` | Run the analysis interactively (generated; not in the repository). |
| `requirements.txt` | Pinned Python dependencies. |
| `static/canvas_utility.html` | Interactive canvas export demo. |
| `capture_canvas_utility.py` | Capture the demo in Brave and Tor. |
| `verify_qr_export.py` | Compare exported QR modules with their reference grid. |
| `tests/` | Regression tests for loading, metrics and probe saving. |

The optional `fingerprintjs/` checkout is needed only when developing or
rebuilding the instrument. Normal collection uses the prebuilt bundle.

## Setup

Use Python 3.12 or newer; this is required by the pinned NumPy dependency.
Run these commands from the repository root:

```bash
cd experiment
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Create the environment at its final location. Moving it can invalidate paths
embedded in its executables.

To collect new observations, install the browsers you intend to test. The driver
currently assumes macOS application paths for Brave and Tor Browser; update
`BRAVE_BINARY` and `TOR_BINARY` in `run_experiment.py` for other locations (for
Linux, see *Running on Ubuntu Linux*).
Selenium also needs compatible browser drivers; initial driver setup may require
network access.

## Run the analysis

**The measurement dataset is not distributed with this repository.** It records
one author's machine, so `data/results.jsonl` is kept local. Without it the
reports below stop with `data/results.jsonl not found`, and the `IncludedDataset`
regression tests are skipped as errors. Collect your own dataset first (see
*Collect observations*), or read the results in `paper4/paper.pdf`.

With a dataset in place, from `experiment/`:

```bash
./.venv/bin/python entropy_budget.py
./.venv/bin/python canvas_mechanism.py --no-plot
```

These reports read `data/results.jsonl`. The notebook is generated rather than
stored, so build it first, then open it:

```bash
./.venv/bin/python build_analysis_nb.py
./.venv/bin/jupyter lab analysis.ipynb
```

Edit `build_analysis_nb.py` when changing notebook content. Regenerating the
notebook overwrites the saved notebook and its execution outputs.

## Collect observations

Add the two test domains to `/etc/hosts`:

```text
127.0.0.1 site-a.test
127.0.0.1 site-b.test
```

The server listens on loopback and uses HTTP. In a first terminal, from
`experiment/`, choose a new dataset filename:

```bash
./.venv/bin/python server.py --port 8000 --results data/collection-001.jsonl
```

In a second terminal, also from `experiment/`:

```bash
./.venv/bin/python run_experiment.py --n 20 --n4 5 --browsers chrome brave brave-aggressive firefox firefox-rfp tor-standard
```

Available analysis configurations are `chrome`, `brave`, `brave-aggressive`,
`firefox`, `firefox-rfp`, and `tor-standard`. The collector also supports
`tor-nojs` for passive observations, which the component analysis excludes.

| Level | Collection loop |
|---|---|
| L1 | Two consecutive raw canvas reads within each probe execution. |
| L2 | One browser session per domain, with `--n` page loads. |
| L3 | `--n` fresh browser sessions per domain, each with one page load. |
| L4 | One browser session visiting both domains for `--n4` pairs. |

Use `--levels 2 3 4` to select collection levels. L1 is embedded in every probe
and has no separate runner. Collection uses visible windows by default; keep
headless runs in a separate dataset because the browser configuration differs.

## Quick collection check

For a smoke test, start the server with a separate, unused output filename:

```bash
./.venv/bin/python server.py --port 8000 --results data/smoke-001.jsonl
```

Then run this in a second terminal:

```bash
./.venv/bin/python run_experiment.py --smoke --browsers chrome
```

If the server is already running on that port, stop it before starting the smoke
server. `--smoke` collects two repetitions for L2/L3 and one L4 pair; it checks
the collection pipeline and does not supply the full six-configuration dataset.

## Dataset handling

The server appends to the selected file. Use a new filename for each collection
and do not append smoke tests to your main `data/results.jsonl`. Run indices
restart at zero and currently do not distinguish separate collection batches.

The existing reports and notebook use `data/results.jsonl` by default; changing
the server's `--results` path does not change their input. To inspect a separate
dataset, pass its path to `fplab.data.load(path)` from Python. Full analysis requires observations for all six configurations.

Dataset-specific exclusions are defined in `fplab/data.py`; review them before
applying the analysis to a new collection. Preserve the raw observations so
failed runs and exclusion decisions remain inspectable.

## Instrumentation

The local FingerprintJS changes expose raw canvas images separately from the
library's stabilized canvas component and record the outcome of querying
`WEBGL_debug_renderer_info`. The probe disables FingerprintJS monitoring.

Keep the source patch and the served bundle synchronized when changing the
instrument. Changes to the optional upstream checkout do not automatically
update `experiment/static/fp.umd.js`.

## Validation and optional canvas demo

From `experiment/`, run the analysis regression tests and the probe-saving test:

```bash
./.venv/bin/python -m unittest discover -s tests -v
node tests/test_probe.cjs
```

The loader prints counts for non-JavaScript observations, exclusions, failed
probes and invalid records. L1 reports valid, failed and unavailable canvas pairs
separately; missing or unreadable images never count as matching images.
Component errors do not erase other usable readings from the same observation.
`data.component_coverage(rows)` reports observed, failed and missing component counts.
Use `data.load(path, return_report=True)` to obtain both rows and audit counts.
Malformed JSON stops loading with the file and line number so it can be inspected.

The optional canvas demo uses `segno` for QR reference generation, not OpenCV;
install it separately with `./.venv/bin/python -m pip install segno` if needed.
With the local server running, use `capture_canvas_utility.py` to capture the demo,
then `verify_qr_export.py` to check the exported module grids. These utilities
are separate from fingerprint collection and do not append to `results.jsonl`.
The two captures, `figures/canvas_utility_brave.png` and
`figures/canvas_utility_tor.png`, are the ones shown in the paper's appendix.

## Running on Ubuntu Linux

The analysis (notebook, `entropy_budget.py`, `canvas_mechanism.py`, tests) runs
unchanged on Linux; collection needs the changes below. These steps target
Ubuntu 24.04, whose Python 3.12 satisfies the pinned NumPy. Ubuntu 22.04 ships
Python 3.10 and needs a newer interpreter.

Install the system packages, then create the environment as in *Setup*. Do not
copy a `.venv` from macOS: it points to a macOS interpreter.

```bash
sudo apt install python3-venv nodejs xvfb
```

Install the browsers from native packages, not snaps:

- **Chrome:** Google's `.deb`. Ubuntu's `chromium` package is a snap.
- **Brave:** `brave-browser` from Brave's APT repository (https://brave.com/linux/).
- **Firefox:** Ubuntu's preinstalled Firefox is a snap, which cannot read the
  profile geckodriver creates in `/tmp` ("Your Firefox profile cannot be
  loaded"). Install Mozilla's `.deb`
  (https://support.mozilla.org/en-US/kb/install-firefox-linux), or set `TMPDIR`
  to a directory under your home before running the collector.
- **Tor Browser:** the Linux tarball from torproject.org, extracted to
  `~/tor-browser`.

Then edit `run_experiment.py`:

1. Point the two binaries at the Linux installs:

   ```python
   BRAVE_BINARY = "/usr/bin/brave-browser"
   TOR_BINARY = os.path.expanduser("~/tor-browser/Browser/firefox")
   ```

2. Give geckodriver Tor Browser's own libraries and font configuration, as
   `start-tor-browser` and tbselenium do. Without them Tor Browser does not load
   its bundled fonts, and font results will not match a real Tor Browser. Pass
   the environment to the Tor driver only, so the Firefox arms keep your `HOME`:

   ```python
   from selenium.webdriver.firefox.service import Service as FirefoxService

   TOR_DIR = os.path.expanduser("~/tor-browser/Browser")
   TOR_ENV = {**os.environ, "HOME": TOR_DIR,
              "PATH": TOR_DIR + os.pathsep + os.environ.get("PATH", ""),
              "LD_LIBRARY_PATH": os.path.join(TOR_DIR, "TorBrowser", "Tor"),
              "FONTCONFIG_PATH": os.path.join(TOR_DIR, "TorBrowser", "Data", "fontconfig"),
              "FONTCONFIG_FILE": "fonts.conf"}
   ```

   In `make_driver`, start Tor with
   `webdriver.Firefox(options=_tor_options(...), service=FirefoxService(env=TOR_ENV))`.

The Brave `excludeSwitches` workaround in `_chromium_options` fixes a macOS-only
crash and is harmless on Linux.

Run the server and collector as in *Collect observations*. Without a desktop
session, for example over SSH, wrap the collector in `xvfb-run -a` rather than
using `--headless`:

```bash
xvfb-run -a ./.venv/bin/python run_experiment.py --n 20 --n4 5 --browsers chrome brave brave-aggressive firefox firefox-rfp tor-standard
```

If Chrome or Brave stops at an "Unlock login keyring" dialog, add
`--password-store=basic` to `_chromium_options`. Do not run the collector as
root; Chromium's sandbox refuses it.

Observations collected on Linux describe a different host from the author's
macOS dataset, so their values differ. Review the exclusions in `fplab/data.py`
before analyzing them.

The paper ships only as the compiled `paper4/paper.pdf`. Its LaTeX sources are
not part of this repository, so there is nothing to build here.

## License

Code: [MIT License](LICENSE) | Paper & Figures: [CC BY 4.0](LICENSE)

`experiment/static/fp.umd.js` redistributes FingerprintJS v5.2.0, which is MIT
licensed by FingerprintJS, Inc.; its notice is reproduced in `LICENSE`.
