# SPDX-License-Identifier: MIT
"""Regression checks run without collecting or changing any observations."""
import base64
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from fplab import data, canvas, entropy, similarity, stability


def png(color):
    stream = io.BytesIO()
    Image.new("RGBA", (2, 2), color).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


A, B = png("red"), png("blue")


def row(**changes):
    r = dict(browser="chrome", level=3, run_index=0, domain=data.DOMAINS[0],
             visitorId="id", components={"canvas": {"value": "skipped"},
                                          "fonts": {"value": ["font"]}},
             level1_canvas_call1=dict(textImage1=A, textImage2=A,
                                      upstreamVerdict="skipped"))
    r.update(changes)
    return r


class InputValidation(unittest.TestCase):
    def test_load_audit_and_explicit_path(self):
        records = [row(), row(error="probe failed"), row(browser="tor-nojs"),
                   [], row(domain="unknown"), row(browser="firefox-rfp", run_index=10),
                   row(level1_canvas_call1={})]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records))
            before = path.read_bytes()
            with patch.object(data, "root", side_effect=AssertionError("default path not needed")):
                rows, audit = data.load(path, verbose=False, return_report=True)
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(rows), 2)
        self.assertEqual(audit, dict(total=7, non_js=1, excluded=1, failed=1, invalid=2, used=2))

    def test_malformed_json_names_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(row()) + "\n{broken\n")
            with self.assertRaisesRegex(ValueError, r"bad.jsonl:2: invalid JSON"):
                data.load(path, verbose=False)

    def test_component_error_is_not_null_or_discarded_row(self):
        r = row(components={"fonts": {"error": "blocked"}, "audio": {"value": 1}})
        self.assertIsNone(data.value(r, "fonts"))
        self.assertEqual(data.value(r, "audio"), "1")
        self.assertIsNone(data.value(r, "missing"))

    def test_successful_undefined_is_observed_unsupported(self):
        r = row(components={"osCpu": {"duration": 0}, "fonts": {"value": None}})
        self.assertEqual(data.value(r, "osCpu"), "null")
        self.assertEqual(data.value(r, "fonts"), "null")

    def test_l1_denominators_and_skipped_raw_images(self):
        rows = [row(), row(level1_canvas_call1=dict(textImage1=A, textImage2=B)),
                row(level1_canvas_call1={}), row(level1_canvas_call1=None),
                row(level1_canvas_call1={"error": "read failed"}),
                row(level1_canvas_call1=dict(textImage1="", textImage2=""))]
        out = stability.level1(rows).set_index("browser").loc["chrome"]
        self.assertEqual((out.n, out.n_valid, out.n_failed, out.n_unavailable), (6, 2, 1, 3))
        self.assertEqual(out.intra_call_match, .5)

    def test_missing_l1_and_corrupt_images_are_unavailable(self):
        r = row()
        del r["level1_canvas_call1"]
        for bad in (r, row(level1_canvas_call1={}),
                    row(level1_canvas_call1=dict(textImage1="data:image/png;base64,!!!",
                                                textImage2="data:image/png;base64,!!!"))):
            self.assertIsNone(canvas.decode(bad))
            self.assertEqual(data.canvas_status(bad), "unavailable")
            self.assertTrue(math.isnan(stability.level1([bad]).iloc[0].intra_call_match))

    def test_empty_image_shapes_are_not_compared(self):
        self.assertIsNone(canvas.compare(None, None))
        self.assertIsNone(canvas.compare(np.zeros((2, 2, 4)), np.zeros((3, 2, 4))))
        self.assertEqual(canvas._pairs_against_first([]), [])

    def test_cross_domain_missing_not_agreement(self):
        rows = [row(level=4, domain=d, components={"fonts": {"error": "failed"}},
                    level1_canvas_call1={}) for d in data.DOMAINS]
        out = stability.cross_domain(rows)["chrome"]
        self.assertTrue(math.isnan(out["component_agreement"]))
        self.assertEqual(out["n_valid_comparisons"], 0)
        self.assertEqual(out["n_unavailable_comparisons"], 1)

    def test_cross_domain_compares_raw_canvas_not_sentinel(self):
        r1, r2 = row(level=4), row(level=4, domain=data.DOMAINS[1])
        r2["level1_canvas_call1"]["textImage1"] = B
        out = stability.cross_domain([r1, r2])["chrome"]
        self.assertEqual(out["differing"], ["canvas"])
        self.assertEqual(out["component_agreement"], .5)

    def test_l4_missing_identifier_not_counted(self):
        rows = [row(level=4, domain=d, visitorId=None) for d in data.DOMAINS]
        self.assertTrue(math.isnan(stability.level4_same(rows)["chrome"]))

    def test_mechanism_never_crosses_domains(self):
        rows = [row(domain=d, run_index=i, level1_canvas_call1=dict(textImage1=im))
                for d, im in zip(data.DOMAINS, (A, B)) for i in range(12)]
        out = canvas.mechanism(rows)["chrome"]
        self.assertEqual(out["n_pairs"], 20)
        self.assertEqual(out["pct_changed"], 0)
        self.assertEqual(canvas.mechanism(rows, max_pairs=3)["chrome"]["n_pairs"], 3)

    def test_similarity_does_not_reward_missing_attributes(self):
        self.assertTrue(math.isnan(similarity.agreement((None,), (None,))))
        self.assertEqual(similarity.agreement(("null",), ("null",)), 1)
        self.assertEqual(similarity.agreement(("a", "b"), ("a", "c")), .5)
        with self.assertRaises(ValueError):
            similarity.agreement(("a",), ("a", "b"))
        with self.assertRaisesRegex(ValueError, "complete observations"):
            similarity.separation([])

    def test_absent_budget_is_unknown_not_protection(self):
        table, totals = entropy.budget([row(components={})])
        self.assertEqual(table[("chrome", "fonts")][0], "ABSENT")
        self.assertTrue(math.isnan(totals["chrome"][0]))
        self.assertTrue(math.isnan(entropy.unweighted_reduction([row(components={})])[0]["chrome"]))

    def test_missing_canvas_properties_display(self):
        self.assertTrue((canvas.properties_frame([]) == "unavailable").all().all())

    def test_component_coverage_preserves_other_readings(self):
        rows = [row(components={"fonts": {"error": "blocked"}}), row()]
        coverage = data.component_coverage(rows).set_index(["browser", "component"])
        self.assertEqual(coverage.loc[("chrome", "fonts"), "n_failed"], 1)
        self.assertEqual(coverage.loc[("chrome", "fonts"), "n_observed"], 1)
        self.assertEqual(coverage.loc[("chrome", "canvas"), "n_unavailable"], 1)


class IncludedDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.audit = data.load(verbose=False, return_report=True)

    def test_coverage_unchanged(self):
        self.assertEqual(self.audit, dict(total=602, non_js=62, excluded=2, failed=0, invalid=0, used=538))
        l1 = stability.level1(self.rows)
        self.assertEqual(l1.n_valid.sum(), 538)
        self.assertEqual(l1.n_failed.sum() + l1.n_unavailable.sum(), 0)
        np.testing.assert_array_equal(l1.intra_call_match, [1, 1, 1, 1, 0, 0])

    def test_headline_budget_unchanged(self):
        totals = entropy.budget(self.rows)[1]
        np.testing.assert_allclose([totals[a][0] for a in data.ARMS],
                                   [48.65, 33.14, 33.14, 48.65, 28.73, 4.65])

    def test_sensitivity_matches_independent_arithmetic(self):
        frame = entropy.classification_sensitivity(self.rows)
        self.assertAlmostEqual(frame.loc["Both charged", "tor-standard"],
                               100*(1-(4.65+4.89+8.38)/48.65))
        self.assertAlmostEqual(frame.loc["Screen charged", "firefox-rfp"],
                               100*(1-(28.73+4.89)/48.65))
        np.testing.assert_allclose(frame.brave, 100*(1-33.14/48.65))

    def test_similarity_safe_threshold(self):
        sep = similarity.separation(self.rows)
        self.assertEqual(sep["n_pairs"], 780)
        self.assertEqual(sep["n_attributes"], 43)
        self.assertEqual(sep["n_incomplete"], 0)
        self.assertTrue(np.all(sep["within"] >= .87))
        self.assertTrue(all(np.all(v < .87) for v in sep["others"].values()))
        self.assertEqual(similarity.exact_matches(self.rows, "brave"), (1, 40))

    def test_heatmap_and_l4_use_raw_canvas(self):
        heat = stability.dispositions(self.rows)
        self.assertEqual(heat.loc["canvas", "firefox-rfp"], stability.RANDOMIZED)
        self.assertEqual(heat.loc["canvas", "tor-standard"], stability.RANDOMIZED)
        xd = stability.cross_domain(self.rows)
        for a in ("firefox-rfp", "tor-standard"):
            self.assertAlmostEqual(xd[a]["component_agreement"], 41/42)
            self.assertEqual(xd[a]["differing"], ["canvas"])

    def test_matched_domain_canvas_results(self):
        mech = canvas.mechanism(self.rows)
        self.assertEqual(mech["brave"]["n_pairs"], 20)
        self.assertAlmostEqual(mech["brave"]["pct_changed"], 3.58125)
        self.assertEqual(mech["brave"]["max_delta"], 1)
        self.assertEqual(sum(mech["brave"]["hist"].values()), 10314)
        self.assertEqual(mech["firefox-rfp"]["max_delta"], 252)
        self.assertEqual(mech["tor-standard"]["max_delta"], 236)


class CollectionIntegration(unittest.TestCase):
    def test_server_acknowledges_saved_rows_and_keeps_demo_separate(self):
        import server
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "observations.jsonl"
            with patch.object(server, "RESULTS_PATH", str(output)):
                client = server.app.test_client()
                with client.get("/canvas_utility.html") as response:
                    self.assertEqual(response.status_code, 200)
                self.assertFalse(output.exists(), "demo route must not collect observations")
                self.assertEqual(client.post("/collect", json=row()).status_code, 200)
                self.assertEqual(client.post("/collect", json=row(error="agent failure")).status_code, 200)
                rows, audit = data.load(output, verbose=False, return_report=True)
                self.assertEqual(len(rows), 1)
                self.assertEqual(audit["failed"], 1)
                self.assertEqual(len(output.read_text().splitlines()), 2)
                self.assertIn("_server_ts", rows[0])


if __name__ == "__main__":
    unittest.main()
