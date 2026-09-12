# SPDX-License-Identifier: MIT
"""Regression checks for the paper-layout figures (no private dataset needed)."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fplab import data, entropy, figures


class PaperFiguresTests(unittest.TestCase):
    def setUp(self):
        self.cap = {
            arm: dict(bits=28.73 if arm == 'firefox-rfp' else 33.14,
                      population=entropy.POPULATION[arm],
                      k={d: entropy.anonymity_set(33.14, entropy.POPULATION[arm], d)
                         for d in (1., 2.)})
            for arm in data.ARMS
        }

    def tearDown(self):
        plt.close('all')

    def test_rfp_scenario_and_no_input_mutation(self):
        before = copy.deepcopy(self.cap)
        with patch.object(figures, '_save', side_effect=lambda fig, *a, **kw: fig):
            fig = figures.anonymity_sets(self.cap)
        ax = fig.axes[0]
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()],
                         ['Chrome', 'Brave', 'Firefox', 'Firefox RFP', 'Tor'])
        rfp_line = next(line for line in ax.lines
                        if len(line.get_ydata()) == 3 and list(line.get_ydata()) == [3, 3, 3])
        np.testing.assert_allclose(rfp_line.get_xdata(),
                                   [.0022459921011529172, 47.39189910895023, 347.1868678191591])
        self.assertEqual(self.cap, before)
        self.assertIsNone(entropy.POPULATION['firefox-rfp'])
        labels = ' '.join(t.get_text() for t in ax.texts)
        self.assertIn('assumed population = 1 million', labels)
        self.assertNotIn('assumed buckets', labels)

    def test_invalid_population(self):
        for n in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                figures.anonymity_sets(self.cap, rfp_population=n)

    def test_heatmap_preserves_values_and_engine_groups(self):
        values = np.arange(132).reshape(22, 6) % 4
        table = pd.DataFrame(values, columns=data.ARMS)
        with patch.object(figures, '_save', side_effect=lambda fig, *a, **kw: fig):
            fig = figures.component_heatmap(table[table.columns[::-1]])
        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.images[0].get_array(), values)
        labels = [t.get_text() for t in ax.texts]
        for label in ('Chromium engine', 'Gecko engine', 'Baseline: Chrome', 'Baseline: Firefox'):
            self.assertIn(label, labels)

    def test_png_and_pdf_exports(self):
        table = pd.DataFrame(np.zeros((22, 6)), columns=data.ARMS)
        with tempfile.TemporaryDirectory() as tmp, patch.object(figures, 'figdir', return_value=Path(tmp)):
            for path in (figures.anonymity_sets(self.cap), figures.component_heatmap(table)):
                self.assertGreater(path.stat().st_size, 1000)
                self.assertGreater(path.with_suffix('.pdf').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
