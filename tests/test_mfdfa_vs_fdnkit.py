"""The batched estimator must agree with fdnkit, the published reference, on every case that matters.

If this file fails, every dynamics feature in the paper is suspect, because the paper's numbers are
computed with `batch_dfa` and its validity is borrowed entirely from fdnkit's validation against
analytic truth. The cases are chosen where a vectorised rewrite most plausibly diverges:

* well-behaved signals across the alpha range (fGn H = 0.3..0.9, fBm, white noise);
* constant runs, where the scale-relative floor actually binds and negative-q moments are dominated
  by the floored segments -- the case project NB found breaks most implementations;
* scales longer than the window, which fdnkit drops as empty;
* rel_floor = 0, the unguarded path;
* detrending orders 1-3, where conditioning differs (polyfit on raw t vs QR on centred t).

Tolerance: QR projection and polyfit's least squares agree to rounding, not bit-for-bit. 1e-9 on
exponents is ~7 orders of magnitude below the smallest effect this project could report.
"""
import warnings

import numpy as np
import pytest
from fdnkit.dfa import dfa  # noqa: E402
from fdnkit.mfdfa import mfdfa  # noqa: E402
from fdnkit.synthetic import fbm, fgn  # noqa: E402

from ampladder.mfdfa import batch_dfa, batch_mfdfa, longest_flat_run, segment_rms  # noqa: E402

TOL = 1e-9
Q = np.array([-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0])
N = 2048
SCALES = np.unique(np.round(np.geomspace(8, N // 8, 12)).astype(int))


def _windows():
    rng = np.random.default_rng(0)
    out = [fgn(N, hurst=h, seed=10 + i) for i, h in enumerate((0.3, 0.5, 0.7, 0.9))]
    out += [fbm(N, hurst=h, seed=20 + i) for i, h in enumerate((0.3, 0.7))]
    out.append(rng.standard_normal(N))
    # Realistic amplitude: sEEG in volts is ~1e-5, the regime that broke neurokit2 in project NB.
    out.append(1e-5 * fgn(N, hurst=0.6, seed=31))
    return np.vstack(out)


def _with_flat_runs():
    x = fgn(N, hurst=0.7, seed=41)
    y = x.copy()
    y[300:364] = y[300]          # a 64-sample saturation plateau
    z = x.copy()
    z[1000:1032] = 0.0           # a 32-sample dropout
    z[1500:1600] = z[1500]       # a 100-sample clip
    return np.vstack([y, z])


def _reference(x, scales, order, rel_floor):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = mfdfa(x, scales=scales, q=Q, order=order, rel_floor=rel_floor)
    d = dfa(x, scales=scales, order=order, rel_floor=rel_floor)
    return d.hurst, m.hurst, np.asarray(m.hq), np.asarray(m.fluct), np.asarray(m.fluct_q)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("rel_floor", [1e-3, 0.0])
def test_matches_fdnkit_on_clean_signals(order, rel_floor):
    X = _windows()
    scales = SCALES[SCALES >= order + 2]
    got = batch_mfdfa(X, scales, q=Q, order=order, rel_floor=rel_floor)
    got_dfa = batch_dfa(X, scales, order=order, rel_floor=rel_floor)
    for i, x in enumerate(X):
        h_dfa, h_m, hq, fl, flq = _reference(x, scales, order, rel_floor)
        assert abs(got_dfa[i] - h_dfa) < TOL
        assert abs(got["hurst"][i] - h_m) < TOL
        np.testing.assert_allclose(got["hq"][i], hq, atol=TOL, rtol=0)
        np.testing.assert_allclose(got["fluct"][i], fl, rtol=1e-9, atol=0)
        np.testing.assert_allclose(got["fluct_q"][i], flq, rtol=1e-8, atol=0)


def test_matches_fdnkit_on_plateau_windows_at_the_pipelines_floor():
    """At rel_floor = 1e-3 (the setting this project uses) the floor binds on every segment inside
    a plateau, so both implementations see the same floored value and agree on every q.

    Not to 1e-9, though. A segment whose raw residual is pure rounding is *excluded* from the
    median that sets the floor when polyfit happens to round it to exactly 0.0, and included when
    the QR path rounds it to 1e-15. That moves the median by up to one rank position (0.07% on the
    worst scale here), and the floored segments carry it into every moment.

    Measured (2026-10-01): max |difference| 6.7e-5 for q <= 0, where the floored segments dominate,
    and 3.4e-9 for q > 0, where they barely register. Tolerances are set just above those. Both are
    irrelevant in practice, because plateau windows are excluded from negative-q features
    upstream -- see the next test for why they must be.
    """
    X = _with_flat_runs()
    got = batch_mfdfa(X, SCALES, q=Q, rel_floor=1e-3)
    neg, pos = Q <= 0, Q > 0
    for i, x in enumerate(X):
        _, h_m, hq, _, _ = _reference(x, SCALES, 1, 1e-3)
        assert abs(got["hurst"][i] - h_m) < TOL
        np.testing.assert_allclose(got["hq"][i][pos], hq[pos], atol=1e-8, rtol=0)
        np.testing.assert_allclose(got["hq"][i][neg], hq[neg], atol=1e-4, rtol=0)


def test_plateaus_inflate_multifractal_width_in_both_implementations():
    """Why plateau windows are excluded upstream rather than trusted to the floor.

    Every window here is monofractal, so the true width h(-5) - h(5) is zero and anything else is
    artefact. Clean windows at N = 2048 show the expected finite-size width (measured 0.08-0.27).
    A single plateau raises it past 2 *with* fdnkit's floor, and past 10 without it -- where the
    value is then set by floating-point rounding and differs between implementations. The floor
    makes the number reproducible; it does not make it meaningful.
    """
    clean = batch_mfdfa(_windows(), SCALES, q=Q, rel_floor=1e-3)["hq"]
    plateau = batch_mfdfa(_with_flat_runs(), SCALES, q=Q, rel_floor=1e-3)["hq"]
    unfloored = batch_mfdfa(_with_flat_runs(), SCALES, q=Q, rel_floor=0.0)["hq"]
    width = lambda h: h[:, 0] - h[:, -1]  # noqa: E731
    assert width(clean).max() < 0.5
    assert width(plateau).min() > 2.0
    assert width(unfloored).min() > 10.0
    for x in _with_flat_runs():
        _, _, hq, _, _ = _reference(x, SCALES, 1, 1e-3)
        assert hq[0] - hq[-1] > 2.0, "fdnkit no longer inflates width on plateaus; revisit the QC"


def test_flat_run_qc_flags_every_plateau_window_and_matches_fdnkit():
    from fdnkit.preprocessing import find_flat_runs

    X = np.vstack([_windows(), _with_flat_runs()])
    got = longest_flat_run(X)
    for i, x in enumerate(X):
        runs = find_flat_runs(x, min_length=2)
        ref = int((runs[:, 1] - runs[:, 0]).max()) if runs.size else 1
        assert got[i] == ref
    smallest = int(SCALES.min())
    flagged = got >= smallest
    assert flagged[-2:].all(), "both plateau windows must be flagged"
    assert not flagged[:-2].any(), "no clean window may be flagged"


def test_floor_actually_binds_in_the_flat_case():
    """Guard the guard: if the floor never binds, the previous test proves nothing about it."""
    X = _with_flat_runs()
    floored = segment_rms(X, SCALES, rel_floor=1e-3)
    raw = segment_rms(X, SCALES, rel_floor=0.0)
    changed = sum(int(np.sum(a != b)) for a, b in zip(floored, raw))
    assert changed > 0


def test_scales_longer_than_window_are_dropped_like_fdnkit():
    x = fgn(256, hurst=0.7, seed=51)[None, :]
    scales = np.array([8, 16, 32, 64, 128, 512])   # 512 > 256
    got = batch_mfdfa(x, scales, q=Q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = mfdfa(x[0], scales=scales, q=Q)
    assert np.isnan(got["fluct"][0, -1]) and np.isnan(ref.fluct[-1])
    assert abs(got["hurst"][0] - ref.hurst) < TOL
    np.testing.assert_allclose(got["hq"][0], ref.hq, atol=TOL, rtol=0)


def test_non_finite_input_fails_loudly():
    X = _windows()
    X[2, 100] = np.nan
    with pytest.raises(ValueError):
        batch_mfdfa(X, SCALES, q=Q)


def test_scale_below_order_plus_two_is_rejected():
    with pytest.raises(ValueError):
        batch_mfdfa(_windows(), np.array([3, 8, 16]), q=Q, order=2)


def test_amplitude_invariance():
    """h(q) must not depend on units. Project NB found neurokit2 violates this on volt-scale sEEG;
    the reason it matters here is that BrainBERT/PopT z-score per window and the linear baseline
    z-scores per training set -- a dynamics feature must be indifferent to both."""
    X = _windows()
    a = batch_mfdfa(X, SCALES, q=Q)["hq"]
    b = batch_mfdfa(X * 3.7e4, SCALES, q=Q)["hq"]
    c = batch_mfdfa((X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True), SCALES, q=Q)["hq"]
    np.testing.assert_allclose(a, b, atol=1e-9, rtol=0)
    np.testing.assert_allclose(a, c, atol=1e-9, rtol=0)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("which", ["clean", "plateau"])
def test_torch_backend_equals_numpy_backend(order, which):
    """The GPU path (used for sensitivity analyses S1/S3) must equal the validated CPU path."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    X = _windows() if which == "clean" else _with_flat_runs()
    sc = np.array([8, 16, 32, 64, 128, 256, 512])
    a = batch_mfdfa(X, sc, q=Q, order=order)
    b = batch_mfdfa(X, sc, q=Q, order=order, backend="torch")
    np.testing.assert_allclose(b["hq"], a["hq"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(b["fluct"], a["fluct"], rtol=1e-10, atol=0)
