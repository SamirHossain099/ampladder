"""The package must compute exactly what the paper's code computed.

The paper's numbers come from the analysis scripts (../analysis). The package is a copy arranged for
reuse; if the two ever diverge, a user of the package would no longer be computing what the paper
reports. These tests run both on the same inputs and require identical outputs.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_SRC = os.path.join(os.path.dirname(__file__), "..", "analysis")
if not os.path.isdir(PROJECT_SRC):
    pytest.skip("project sources not present (installed package only)", allow_module_level=True)
sys.path.insert(0, PROJECT_SRC)

from ampladder import foldaudit, ladder, mfdfa, selection  # noqa: E402

import batch_dfa as paper_mfdfa  # noqa: E402
import d1_selection as paper_selection  # noqa: E402
import d3_common_time_index as paper_d3  # noqa: E402


def test_mfdfa_identical_to_paper():
    rng = np.random.default_rng(5)
    X = np.cumsum(rng.standard_normal((40, 2048)), axis=1)
    sc = ladder.default_scales(2048)
    a = paper_mfdfa.batch_mfdfa(X, sc, q=ladder.Q_POSITIVE)["hq"]
    b = mfdfa.batch_mfdfa(X, sc, q=ladder.Q_POSITIVE)["hq"]
    assert np.array_equal(a, b)


def test_default_scales_are_the_paper_scales():
    assert np.array_equal(ladder.default_scales(2048), np.unique(np.round(np.geomspace(8, 256, 12)).astype(int)))


def test_selection_identical_to_paper():
    labels = [f"A{i}" for i in range(1, 9)] + [f"B{i}" for i in range(1, 6)] + ["C1"]
    rng = np.random.default_rng(1)
    scores = {e: float(rng.uniform(0.49, 0.53)) for e in labels}
    assert (selection.select_electrodes_from_scores(labels, scores, 8)
            == paper_selection.select_electrodes_from_scores(labels, scores, 8))


def test_common_time_folds_identical_to_paper():
    rng = np.random.default_rng(2)
    items = pd.DataFrame(dict(base_index=np.arange(300), window_from=rng.permutation(300) * 2500,
                              label=rng.integers(0, 2, 300)))
    a = paper_d3.common_time_folds(items)
    b = foldaudit.common_time_folds(items)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
