"""Extract the ladder's features for one Neuroprobe Lite session, streaming one electrode at a time.

Reads exactly the windows listed in `results/index/neuroprobe_lite_index.csv.gz` (Neuroprobe's own
windows), so no window boundary is re-implemented here.

Faithfulness to the published pipeline, by construction
-------------------------------------------------------
`eval_population.py` feeds each item, as a float32 torch tensor of shape (1, electrodes, 2048), to
`preprocess_data(..., 'laplacian-stft_abs', ...)`: Neuroprobe's Laplacian, then Neuroprobe's STFT.
This script calls those same two functions, in that order, on float32 torch tensors, in batches.
L5 is their output; L4 is L5 divided by its own sum per (window, electrode); L1 and L2 are computed
from the same float32 Laplacian output (cast to float64 for the arithmetic). `tests/test_extract.py`
checks batched output against their per-item call.

Memory and disk
---------------
One electrode's trace is about 171 MB as float64; the session never sits in RAM. Windows go to a
scratch float32 array on D: (deleted at the end), then features are computed in batches.

Outputs, in --out (D:/ieeg07/features by default), per session `s{subject}_t{trial}`:
  *_meta.json        electrode labels (Lite order), window_from per row, settings, counts
  *_window_from.npy  (n_windows,) int64, sorted; row i of every array below is this window
  *_L1.npy           (n_windows, n_electrodes) float64   log variance of the Laplacian window
  *_hq.npy           (n_windows, n_electrodes, 8) float64 h(q), q = -3,-2,-1,0,1,2,3,5
  *_flat.npy         (n_windows, n_electrodes) int32     longest constant run, samples
  *_L5.npy           (n_windows, n_electrodes, 17, 38) float32  Neuroprobe STFT magnitude
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from batch_dfa import batch_mfdfa, longest_flat_run  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2048
Q_ALL = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0])
SCALES = np.unique(np.round(np.geomspace(8, N // 8, 12)).astype(int))   # PREREGISTRATION.md s4
STFT = {"stft": {"nperseg": 512, "poverlap": 0.75, "window": "hann",
                 "max_frequency": 150, "min_frequency": 0}}


def neuroprobe_imports(np_root, data_root):
    os.environ["ROOT_DIR_BRAINTREEBANK"] = data_root
    sys.path.insert(0, os.path.join(np_root, "examples"))
    sys.path.insert(0, np_root)
    import torch
    from eval_utils import laplacian_rereference_neural_data, preprocess_stft
    from neuroprobe.braintreebank_subject import BrainTreebankSubject
    from neuroprobe.config import NEUROPROBE_LITE_ELECTRODES
    return torch, laplacian_rereference_neural_data, preprocess_stft, BrainTreebankSubject, \
        NEUROPROBE_LITE_ELECTRODES


def session_windows(index_path, subject, trial, splits=None):
    cols = ["split", "src_subject", "src_trial", "window_from", "window_to"]
    df = pd.read_csv(index_path, usecols=cols)
    df = df[(df.src_subject == subject) & (df.src_trial == trial)]
    if splits:
        df = df[df.split.isin(splits)]
    if df.empty:
        raise ValueError(f"no windows for session {subject}/{trial} in {index_path}")
    if not (df.window_to - df.window_from == N).all():
        raise ValueError("index contains a window that is not exactly 2048 samples")
    return np.sort(df.window_from.unique()).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--np-root", default="D:/ieeg07/np")
    ap.add_argument("--data-root", default="D:/ieeg07/braintreebank")
    ap.add_argument("--index", default=os.path.join(HERE, "results", "index", "neuroprobe_lite_index.csv.gz"))
    ap.add_argument("--out", default="D:/ieeg07/features")
    ap.add_argument("--scratch", default="D:/ieeg07/scratch")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--max-windows", type=int, default=None, help="testing only")
    ap.add_argument("--electrodes-json", default=None,
                    help="D1: JSON {\"s{S}_t{T}\": [labels]} replacing the Lite list for this session")
    ap.add_argument("--splits", nargs="*", default=None, help="restrict windows to these splits")
    a = ap.parse_args()

    torch, laplacian, stft, Subject, LITE_ELEC = neuroprobe_imports(a.np_root, a.data_root)
    tag = f"s{a.subject}_t{a.trial}"
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.scratch, exist_ok=True)

    subj = Subject(a.subject, allow_corrupted=False, cache=False)
    if a.electrodes_json:
        wanted = json.load(open(a.electrodes_json, encoding="utf-8"))[tag]
        unknown = [e for e in wanted if e not in subj.electrode_labels]
        if unknown:
            raise ValueError(f"{tag}: electrodes not available in Neuroprobe's subject: {unknown}")
        labels = list(wanted)
    else:
        labels = [e for e in LITE_ELEC[subj.subject_identifier] if e in subj.electrode_labels]
    h5_keys = [subj.h5_neural_data_keys[e] for e in labels]
    wf = session_windows(a.index, a.subject, a.trial, a.splits)
    if a.max_windows:
        wf = wf[: a.max_windows]
    nw, ne = len(wf), len(labels)
    print(f"{tag}: {nw} windows x {ne} electrodes", flush=True)

    win_path = os.path.join(a.scratch, f"{tag}_windows.f32")
    win = np.memmap(win_path, dtype=np.float32, mode="w+", shape=(nw, ne, N))
    h5_path = os.path.join(a.data_root, f"sub_{a.subject}_trial{a.trial:03}.h5")
    t0 = time.time()
    with h5py.File(h5_path, "r") as f:
        length = f["data"][h5_keys[0]].shape[0]
        if wf.max() + N > length:
            raise ValueError("a window runs past the end of the recording")
        # Peak per electrode: the float64 trace, its float32 copy, the int64 index grid and the
        # float32 window block. Refuse up front rather than thrash.
        resources.require_ram(length * (8 + 4) + nw * N * (8 + 4), "one electrode plus its windows")
        idx = wf[:, None] + np.arange(N)[None, :]
        for j, key in enumerate(h5_keys):
            trace = f["data"][key][:]                      # float64, whole electrode
            # Neuroprobe casts to float32 when it builds the tensor; do the same, then slice.
            win[:, j, :] = trace.astype(np.float32)[idx]
            if (j + 1) % 20 == 0:
                print(f"  read {j + 1}/{ne} electrodes ({time.time() - t0:.0f} s)", flush=True)
    win.flush()

    L1 = np.empty((nw, ne), dtype=np.float64)
    HQ = np.empty((nw, ne, len(Q_ALL)), dtype=np.float64)
    FL = np.empty((nw, ne), dtype=np.int32)
    L5 = np.lib.format.open_memmap(os.path.join(a.out, f"{tag}_L5.npy"), mode="w+",
                                   dtype=np.float32, shape=(nw, ne, 17, 38))
    n_zero_var = 0
    t1 = time.time()
    for b0 in range(0, nw, a.batch):
        b1 = min(nw, b0 + a.batch)
        # An explicit copy: ascontiguousarray on an already-contiguous memmap slice returns a VIEW,
        # and a tensor holding that view keeps the map open, so Windows refuses to delete it.
        x = torch.from_numpy(np.array(win[b0:b1], dtype=np.float32, copy=True))   # (b, ne, N)
        lap, _, _ = laplacian(x, labels, remove_non_laplacian=False)    # Neuroprobe's own
        spec = stft(lap, preprocess="stft_abs", preprocess_parameters=STFT)
        spec = spec.numpy() if hasattr(spec, "numpy") else spec
        if spec.shape[1:] != (ne, 17, 38):
            raise ValueError(f"unexpected STFT shape {spec.shape}")
        L5[b0:b1] = spec
        lap64 = lap.numpy().astype(np.float64).reshape(-1, N)
        var = lap64.var(axis=1)
        zero = var <= 0
        n_zero_var += int(zero.sum())
        with np.errstate(divide="ignore"):
            L1[b0:b1] = np.log(np.where(zero, np.nan, var)).reshape(b1 - b0, ne)
        # A zero-variance electrode-window has no defined exponent: leave it NaN and count it.
        hq = np.full((lap64.shape[0], len(Q_ALL)), np.nan)
        ok = ~zero
        if ok.any():
            hq[ok] = batch_mfdfa(lap64[ok], SCALES, q=Q_ALL)["hq"]
        HQ[b0:b1] = hq.reshape(b1 - b0, ne, len(Q_ALL))
        FL[b0:b1] = longest_flat_run(lap64).reshape(b1 - b0, ne)
        if (b0 // a.batch) % 20 == 0:
            done = b1 / nw
            print(f"  features {b1}/{nw} ({time.time() - t1:.0f} s, eta {(time.time() - t1) / done * (1 - done):.0f} s)",
                  flush=True)
    L5.flush()
    del x, lap, spec, lap64
    win._mmap.close()
    L5._mmap.close()
    del L5, win

    np.save(os.path.join(a.out, f"{tag}_window_from.npy"), wf)
    np.save(os.path.join(a.out, f"{tag}_L1.npy"), L1)
    np.save(os.path.join(a.out, f"{tag}_hq.npy"), HQ)
    np.save(os.path.join(a.out, f"{tag}_flat.npy"), FL)
    meta = dict(subject=a.subject, trial=a.trial, n_windows=nw, n_electrodes=ne,
                electrode_labels=labels, h5_keys=h5_keys, scales=SCALES.tolist(), q=Q_ALL.tolist(),
                stft=STFT, zero_variance_electrode_windows=n_zero_var,
                flat_ge_smallest_scale=int((FL >= SCALES.min()).sum()),
                electrode_windows=int(nw * ne), seconds=round(time.time() - t0, 1))
    with open(os.path.join(a.out, f"{tag}_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    os.remove(win_path)
    print(f"{tag}: done in {meta['seconds']} s; zero-variance {n_zero_var}; "
          f"flat>={SCALES.min()}: {meta['flat_ge_smallest_scale']} of {nw * ne}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
