"""foldaudit on constructed indices whose answers are known by design."""
import numpy as np
import pandas as pd
import pytest

from ampladder.foldaudit import common_time_folds, single_class_share, train_test_adjacency


def _index(test1, test0, train):
    rows = [dict(test_subject=1, test_trial=0, task="t", fold=0, role="test", label=1, window_from=x) for x in test1]
    rows += [dict(test_subject=1, test_trial=0, task="t", fold=0, role="test", label=0, window_from=x) for x in test0]
    rows += [dict(test_subject=1, test_trial=0, task="t", fold=0, role="train", label=0, window_from=x) for x in train]
    return pd.DataFrame(rows)


def test_single_class_share_counts_items_outside_the_shared_span():
    # class 1 at 0..9, class 0 at 5..14: shared span 5..9; items 0-4 (class 1) and 10-14 (class 0) lie outside.
    idx = _index(list(range(10)), list(range(5, 15)), [100])
    assert single_class_share(idx).single_class_share.iloc[0] == pytest.approx(10 / 20)


def test_single_class_share_is_zero_when_classes_interleave():
    idx = _index(list(range(0, 20, 2)), list(range(1, 21, 2)), [100])
    # Only the very first and very last item can lie outside the shared span.
    assert single_class_share(idx).single_class_share.iloc[0] <= 2 / 20


def test_adjacency_uses_seconds():
    fs = 2048.0
    idx = _index([10 * fs], [50 * fs], [10.5 * fs, 200 * fs])
    a = train_test_adjacency(idx, within_seconds=(1.0, 60.0), fs=fs).iloc[0]
    assert a["within_1s"] == pytest.approx(0.5)      # the item at 10 s has a train item 0.5 s away
    assert a["within_60s"] == pytest.approx(1.0)


def test_common_time_folds_are_time_disjoint_and_cover_every_item():
    rng = np.random.default_rng(0)
    items = pd.DataFrame(dict(base_index=np.arange(200), window_from=rng.permutation(200) * 3000,
                              label=rng.integers(0, 2, 200)))
    f = common_time_folds(items)
    for k in (0, 1):
        tr, te = f[(f.fold == k) & (f.role == "train")], f[(f.fold == k) & (f.role == "test")]
        assert te.window_from.min() > tr.window_from.max() or te.window_from.max() < tr.window_from.min()
        assert not set(tr.base_index) & set(te.base_index)
    assert f[f.fold == 0].base_index.nunique() == 200
