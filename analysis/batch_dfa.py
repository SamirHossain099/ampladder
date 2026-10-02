"""Batched DFA / MFDFA over many equal-length windows, numerically matched to fdnkit.

Why this exists
---------------
fdnkit's reference implementation loops over segments with `np.polyfit`, about 20 ms per call on a
2048-sample window (results/known_truth_summary.json). Neuroprobe Lite is roughly 12 sessions x
thousands of word-onset windows x up to 120 electrodes -- on the order of 10^7 windows, which at
that speed is over a hundred single-core hours.

Every window has the same length and every window is analysed on the same scale grid, so the
polynomial detrend at scale `s` is one fixed linear operator: the residual of a least-squares fit on
the design matrix V (Vandermonde in t = 0..s-1) is `seg - seg @ Q @ Q.T`, where `V = QR`. That turns
the inner loop into a matmul over all segments of all windows at once.

What is matched exactly, line by line, against fdnkit `mfdfa._fluctuations` / `mfdfa.mfdfa`
-------------------------------------------------------------------------------------------
* profile: `cumsum(x - mean(x))`;
* forward, non-overlapping partition, `n // s` segments, remainder discarded;
* per-segment RMS of the order-`m` polynomial residual;
* floor: `max(rel_floor * median(positive RMS at that scale), eps)`, applied per window per scale;
* `F(s) = sqrt(mean(RMS^2))`; `F_q(s) = mean(RMS^q)^(1/q)`, `F_0(s) = exp(0.5 mean(log RMS^2))`;
* slopes: OLS of `log2(F + eps)` on `log2(s)` over finite points.

`tests/test_batch_dfa.py` asserts agreement with fdnkit itself -- on fGn, fBm, white noise, signals
with constant runs (where the floor binds) and signals with scales longer than the window -- so
this file cannot drift from the reference without a test failing. The QR projection and polyfit's
scaled-Vandermonde least squares agree to rounding error, not bit-for-bit, so the tolerance is
stated in the test rather than implied.
"""
from __future__ import annotations

import numpy as np

EPS = np.finfo(float).eps


def _projector(s: int, order: int) -> np.ndarray:
    """Orthonormal basis Q (s x (order+1)) of the polynomial design at scale s."""
    t = np.arange(s, dtype=float)
    # Centre and scale t before building powers: the column space is unchanged (so the residual is
    # identical in exact arithmetic) but the Vandermonde stays well conditioned at s = 256+.
    tc = (t - t.mean()) / max(1.0, t.std())
    v = np.vander(tc, order + 1, increasing=True)
    q, _ = np.linalg.qr(v)
    return q


def validate_scales(scales, order: int) -> np.ndarray:
    scales = np.asarray(scales, dtype=int)
    if scales.ndim != 1 or scales.size == 0:
        raise ValueError("scales must be a non-empty 1-D sequence")
    if np.any(scales < order + 2):
        raise ValueError(f"every scale must be at least order + 2 = {order + 2}")
    return scales


def segment_rms(x: np.ndarray, scales, order: int = 1, rel_floor: float = 1e-3):
    """Per-window, per-scale arrays of floored segment RMS.

    x : (B, n) array of windows. Returns a list over scales; entry i is (B, n // s_i), or an empty
    (B, 0) array when the scale exceeds the window, mirroring fdnkit's empty-array convention.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be 2-D: (n_windows, n_samples)")
    if not np.all(np.isfinite(x)):
        # fdnkit would propagate NaN silently (a known open item in its backlog). Here it is a
        # loud failure: a NaN window must be excluded upstream, with the exclusion counted.
        raise ValueError("non-finite values in input windows")
    b, n = x.shape
    scales = validate_scales(scales, order)
    y = np.cumsum(x - x.mean(axis=1, keepdims=True), axis=1)
    out = []
    for s in scales:
        s = int(s)
        k = n // s
        if k == 0:
            out.append(np.empty((b, 0)))
            continue
        seg = y[:, : k * s].reshape(b, k, s)
        q = _projector(s, order)
        resid = seg - (seg @ q) @ q.T
        rms = np.sqrt(np.mean(resid * resid, axis=2))           # (b, k)
        pos = np.where(rms > 0, rms, np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(pos, axis=1)                     # (b,)
        floor = np.where(np.isfinite(med), rel_floor * med, EPS)
        floor = np.maximum(floor, EPS)
        out.append(np.maximum(rms, floor[:, None]))
    return out


def segment_rms_torch(x, scales, order: int = 1, rel_floor: float = 1e-3, device="cuda"):
    """The same computation as `segment_rms`, in torch float64, for the sensitivity analyses.

    Added 2026-10-01 for S1/S3, which recompute the battery under five more settings: on the CPU that
    is about 11 hours, and the computation is memory-bound, which is what a GPU is for. It is held to
    the CPU version (and through it to fdnkit) by `tests/test_batch_dfa.py`, in float64, so a result
    may be computed either way. The CPU path stays the default; the primary results used it.
    """
    import torch

    xt = torch.as_tensor(np.asarray(x, dtype=np.float64), device=device)
    if not torch.isfinite(xt).all():
        raise ValueError("non-finite values in input windows")
    b, n = xt.shape
    scales = validate_scales(scales, order)
    y = torch.cumsum(xt - xt.mean(dim=1, keepdim=True), dim=1)
    out = []
    for s in scales:
        s = int(s)
        k = n // s
        if k == 0:
            out.append(np.empty((b, 0)))
            continue
        seg = y[:, : k * s].reshape(b, k, s)
        q = torch.as_tensor(_projector(s, order), device=device)
        resid = seg - (seg @ q) @ q.T
        rms = torch.sqrt((resid * resid).mean(dim=2))
        pos = torch.where(rms > 0, rms, torch.full_like(rms, float("nan")))
        # torch.nanmedian returns the LOWER middle value for an even count; numpy averages the two.
        # nanquantile(0.5) interpolates as numpy does, so the floor matches where it binds.
        med = torch.nanquantile(pos, 0.5, dim=1)
        floor = torch.where(torch.isfinite(med), rel_floor * med, torch.full_like(med, EPS))
        floor = torch.clamp(floor, min=EPS)
        out.append(torch.maximum(rms, floor[:, None]).cpu().numpy())
    return out


def _slopes(scales, values):
    """OLS slope of log2(values) on log2(scales), per row, over finite entries.

    values : (..., S). Returns (...,).
    """
    ls = np.log2(np.asarray(scales, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.log2(values)
    good = np.isfinite(lv)
    w = good.astype(float)
    cnt = w.sum(axis=-1)
    lv0 = np.where(good, lv, 0.0)
    xm = (w * ls).sum(axis=-1) / np.where(cnt > 0, cnt, 1)
    ym = lv0.sum(axis=-1) / np.where(cnt > 0, cnt, 1)
    dx = (ls - xm[..., None]) * w
    num = (dx * (lv0 - ym[..., None]) * w).sum(axis=-1)
    den = (dx * dx).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = num / den
    slope = np.where((cnt >= 2) & (den > 0), slope, np.nan)
    return slope


def batch_mfdfa(x, scales, q=None, order: int = 1, rel_floor: float = 1e-3, backend: str = "numpy"):
    """Batched MFDFA. Returns dict of arrays:

    fluct   (B, S)     F(s), the q = 2 fluctuation function
    fluct_q (B, Q, S)  F_q(s)
    hurst   (B,)       slope of F(s)  -- equals fdnkit `dfa(...).hurst` and `mfdfa(...).hurst`
    hq      (B, Q)     generalised Hurst exponents h(q)
    """
    scales = validate_scales(scales, order)
    q = np.array([-5, -3, -2, -1, 0, 1, 2, 3, 5], dtype=float) if q is None else np.asarray(q, float)
    if backend == "numpy":
        rms = segment_rms(x, scales, order=order, rel_floor=rel_floor)
    elif backend == "torch":
        rms = segment_rms_torch(x, scales, order=order, rel_floor=rel_floor)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    b = np.asarray(x).shape[0]
    S, Q = len(scales), len(q)
    fluct = np.full((b, S), np.nan)
    fluct_q = np.full((b, Q, S), np.nan)
    for i, r in enumerate(rms):
        if r.shape[1] == 0:
            continue
        r2 = r * r
        fluct[:, i] = np.sqrt(r2.mean(axis=1))
        for j, qq in enumerate(q):
            if qq == 0:
                fluct_q[:, j, i] = np.exp(0.5 * np.log(r2).mean(axis=1))
            else:
                fluct_q[:, j, i] = np.mean(r ** qq, axis=1) ** (1.0 / qq)
    hurst = _slopes(scales, fluct + EPS)
    hq = _slopes(scales, fluct_q + EPS)
    return dict(fluct=fluct, fluct_q=fluct_q, hurst=hurst, hq=hq, scales=np.asarray(scales), q=q)


def longest_flat_run(x, atol: float = 0.0) -> np.ndarray:
    """Length in samples of the longest constant run in each window (1 when there is none).

    Matches fdnkit `preprocessing.find_flat_runs`: a run is a maximal stretch whose successive
    differences are within `atol` (default exact equality, which is what saturation and dropout
    produce), and its length counts samples, not differences.

    Why the pipeline needs this rather than trusting the floor: a segment lying wholly inside a
    constant run has a detrended residual that is pure floating-point rounding. Negative-q moments
    are then dominated by rounding error -- fdnkit's polyfit and this module's QR projection
    produce different rounding and therefore different h(q < 0), and *both* report h(-5) near 14
    on a window whose true H is 0.7 (tests/test_batch_dfa.py). fdnkit's scale-relative floor
    bounds the divergence but does not make the value meaningful. So negative-q features are
    undefined on any window whose longest run reaches the smallest scale, and such windows are
    excluded from them, with the exclusion counted.
    """
    x = np.asarray(x, dtype=float)
    d = np.abs(np.diff(x, axis=1)) <= atol                       # (B, n-1)
    c = np.cumsum(d, axis=1)
    last_break = np.maximum.accumulate(np.where(~d, c, 0), axis=1)
    run_diffs = (c - last_break).max(axis=1) if d.shape[1] else np.zeros(x.shape[0], int)
    return run_diffs.astype(int) + 1


def batch_dfa(x, scales, order: int = 1, rel_floor: float = 1e-3):
    """Batched monofractal DFA exponent; equals fdnkit `dfa(...).hurst` per row."""
    scales = validate_scales(scales, order)
    rms = segment_rms(x, scales, order=order, rel_floor=rel_floor)
    b = np.asarray(x).shape[0]
    fluct = np.full((b, len(scales)), np.nan)
    for i, r in enumerate(rms):
        if r.shape[1]:
            fluct[:, i] = np.sqrt((r * r).mean(axis=1))
    return _slopes(scales, fluct + EPS)
