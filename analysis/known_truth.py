"""Gate 0: can DFA and MFDFA be estimated at all on a Neuroprobe window?

Every Neuroprobe trial is a 1-second window at 2048 Hz, so 2048 samples per electrode per trial
(Neuroprobe paper, Approach section). Project 04 found the canonical DFA aging effect *reverses
sign* below about 2,000 RR intervals -- a finite-size effect at almost exactly this length. So
before comparing a dynamics battery against foundation models on these windows, the battery has to
be shown to measure something at this length. If it cannot, the comparison is between a model and
noise, and the paper's premise fails here rather than after the GPU work.

Known truth, two families:
  * fractional Gaussian noise (fGn), stationary, DFA alpha = H in (0, 1);
  * fractional Brownian motion (fBm), its cumulative sum, DFA alpha = H + 1 in (1, 2).
Raw sEEG voltage at 2048 Hz is dominated by low-frequency power and commonly gives alpha > 1, so
both regimes matter. Both are monofractal, so the true multifractal width is zero and any width
MFDFA reports is a finite-size artefact.

For each (family, H, N) this records, over R replicates:
  * DFA alpha bias and SD;
  * MFDFA spurious width  Delta h = h(q_min) - h(q_max);
  * wall-clock per call, which is gate 4 (cost) of the brief.

The decision quantity is the **single-window SD** of alpha at N = 2048. A decoder sees one window
per trial, so an effect on alpha smaller than that SD is not decodable from a single window no
matter how good the downstream model is.

Writes results/known_truth.csv and results/known_truth_summary.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fdnkit  # noqa: E402
from fdnkit.dfa import dfa  # noqa: E402
from fdnkit.mfdfa import mfdfa  # noqa: E402
from fdnkit.synthetic import fbm, fgn  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FDNKIT_SRC = os.path.dirname(os.path.dirname(os.path.dirname(fdnkit.__file__)))

HURSTS = (0.3, 0.5, 0.7, 0.9)
LENGTHS = (512, 1024, 2048, 4096, 8192)
NEUROPROBE_N = 2048
Q = np.array([-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0])


def scales_for(n):
    """Log-spaced scales from 8 samples to n/8, the conventional upper limit (n/4 is the hard one).

    Fixed by rule, not tuned: the same rule applies at every length, so length effects are not
    confounded with a changing scale grid.
    """
    hi = max(16, n // 8)
    s = np.unique(np.round(np.geomspace(8, hi, 12)).astype(int))
    return s


def fdnkit_commit():
    try:
        out = subprocess.run(["git", "-C", FDNKIT_SRC, "log", "-1", "--format=%h"],
                             capture_output=True, text=True, timeout=20)
        dirty = subprocess.run(["git", "-C", FDNKIT_SRC, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def one(family, h, n, seed):
    if family == "fgn":
        x = fgn(n, hurst=h, seed=seed)
        alpha_true = h
    else:
        x = fbm(n, hurst=h, seed=seed)
        alpha_true = h + 1.0
    sc = scales_for(n)
    t0 = time.perf_counter()
    a = dfa(x, scales=sc).hurst
    t1 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = mfdfa(x, scales=sc, q=Q)
    t2 = time.perf_counter()
    hq = np.asarray(m.hq, dtype=float)
    return dict(family=family, H=h, N=n, seed=seed, alpha_true=alpha_true, alpha=a,
                alpha_err=a - alpha_true, delta_h=float(hq[0] - hq[-1]), h2=float(hq[Q == 2][0]),
                t_dfa_ms=(t1 - t0) * 1e3, t_mfdfa_ms=(t2 - t1) * 1e3)


def summarise(df):
    g = df.groupby(["family", "H", "N"])
    s = g.agg(alpha_true=("alpha_true", "first"), bias=("alpha_err", "mean"),
              sd=("alpha", "std"), rmse=("alpha_err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
              delta_h_median=("delta_h", "median"), delta_h_p95=("delta_h", lambda v: float(np.quantile(v, .95))),
              t_dfa_ms=("t_dfa_ms", "median"), t_mfdfa_ms=("t_mfdfa_ms", "median")).reset_index()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for fam in ("fgn", "fbm"):
        for h in HURSTS:
            for n in LENGTHS:
                for r in range(a.reps):
                    # Seed is a pure function of the cell, never hash(): project 02 and 08 both
                    # lost reproducibility to Python's process-randomised hash.
                    seed = 1_000_000 * (fam == "fbm") + 10_000 * int(h * 10) + 10 * n + r
                    rows.append(one(fam, h, n, seed))
        print(f"{fam} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "known_truth.csv"), index=False)
    s = summarise(df)

    at = s[s.N == NEUROPROBE_N]
    summary = dict(
        reps=a.reps, lengths=list(LENGTHS), hursts=list(HURSTS), q=Q.tolist(),
        scale_rule="12 log-spaced scales from 8 to N/8, same rule at every N",
        fdnkit_version=fdnkit.__version__, fdnkit_commit=fdnkit_commit(),
        python=platform.python_version(), numpy=np.__version__,
        at_neuroprobe_length=at.to_dict(orient="records"),
        alpha_sd_at_2048_max=float(at.sd.max()), alpha_sd_at_2048_min=float(at.sd.min()),
        abs_bias_at_2048_max=float(at.bias.abs().max()),
        spurious_delta_h_median_at_2048=float(at.delta_h_median.median()),
        t_dfa_ms_median_at_2048=float(at.t_dfa_ms.median()),
        t_mfdfa_ms_median_at_2048=float(at.t_mfdfa_ms.median()),
        table=s.to_dict(orient="records"),
    )
    with open(os.path.join(a.out, "known_truth_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    pd.set_option("display.width", 160)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nAt N = {NEUROPROBE_N}: alpha SD {summary['alpha_sd_at_2048_min']:.3f}-"
          f"{summary['alpha_sd_at_2048_max']:.3f}, max |bias| {summary['abs_bias_at_2048_max']:.3f}, "
          f"spurious Delta h median {summary['spurious_delta_h_median_at_2048']:.3f}")
    print(f"cost per call at N = {NEUROPROBE_N}: DFA {summary['t_dfa_ms_median_at_2048']:.2f} ms, "
          f"MFDFA {summary['t_mfdfa_ms_median_at_2048']:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
