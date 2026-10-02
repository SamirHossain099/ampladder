"""The amplitude ladder's feature sets, on plain arrays.

Each function takes windows shaped (n_windows, n_channels, n_samples), already re-referenced the way the
benchmark under study does it, and returns per-channel features. They differ in one property: whether
the window's absolute amplitude survives.

  amplitude_only   log variance per channel                          (only amplitude)
  dynamics         generalized Hurst exponents h(q), q = 1, 2, 3, 5  (amplitude-invariant)
  shape_only       a spectrogram divided by its own total per channel (amplitude removed)

The paper's L5 is the benchmark's own spectrogram, which callers compute with the benchmark's code; L4 is
`shape_only` applied to it. Keeping the benchmark's transform out of this module is deliberate: a
reproduction should call the benchmark's code, not a re-implementation of it.
"""
from __future__ import annotations

import numpy as np

from .mfdfa import batch_mfdfa

N_DEFAULT = 2048
Q_POSITIVE = np.array([1.0, 2.0, 3.0, 5.0])


def default_scales(n: int = N_DEFAULT, s_min: int = 8, n_scales: int = 12) -> np.ndarray:
    """Log-spaced scales from s_min to n/8, the rule used for the paper's primary battery."""
    return np.unique(np.round(np.geomspace(s_min, max(2 * s_min, n // 8), n_scales)).astype(int))


def amplitude_only(windows: np.ndarray) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float64)
    var = x.var(axis=-1)
    if (var <= 0).any():
        raise ValueError("a channel-window has zero variance; amplitude is undefined there")
    return np.log(var)


def dynamics(windows: np.ndarray, scales=None, q=Q_POSITIVE, order: int = 1, backend: str = "numpy") -> np.ndarray:
    x = np.asarray(windows, dtype=np.float64)
    b, c, n = x.shape
    scales = default_scales(n) if scales is None else scales
    hq = batch_mfdfa(x.reshape(b * c, n), scales, q=q, order=order, backend=backend)["hq"]
    return hq.reshape(b, c, len(q))


def shape_only(spectrogram: np.ndarray) -> np.ndarray:
    """Divide each (window, channel) spectrogram (..., time, freq) by its own total."""
    s = np.asarray(spectrogram, dtype=np.float64)
    total = s.sum(axis=(-2, -1), keepdims=True)
    if (total <= 0).any():
        raise ValueError("a channel-window with zero spectral mass cannot be shape-normalized")
    return s / total
