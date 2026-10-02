"""Features for the registered sensitivity analyses S1 and S3 (PREREGISTRATION.md section 8).

S1, estimator settings: the battery (h(q), q = 1, 2, 3, 5) on the Laplacian input under upper scale
  {N/16, N/8, N/4} x order {1, 2}. The primary (N/8, order 1) already exists in D:/ieeg07/features;
  the other five are computed here into D:/ieeg07/features_S1_s{smax}_o{order}/.
S3, input: L1, the battery and L5 on raw voltage (no Laplacian), into D:/ieeg07/features_S3_raw/.

Windows, electrodes, float32 casting and Neuroprobe's own STFT are exactly as in extract_features.py;
only the estimator setting (S1) or the reference (S3) changes. The exponents are computed on the GPU
with `batch_dfa.batch_mfdfa(..., backend="torch")`, which tests hold to the validated CPU path to
1e-9 (tests/test_batch_dfa.py), because the CPU route would take about 11 hours.

Outputs keep extract_features.py's file layout so decode.py runs on them unchanged. S1 directories
hold only the exponents (as the 8-column layout, negative q left NaN and never read by L2).
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

from batch_dfa import batch_mfdfa, longest_flat_run  # noqa: E402
from extract_features import N, Q_ALL, SCALES, STFT, neuroprobe_imports, session_windows  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q_POS = np.array([1.0, 2.0, 3.0, 5.0])
POS_COLS = np.isin(Q_ALL, Q_POS)
S1_SETTINGS = [(128, 1), (512, 1), (128, 2), (256, 2), (512, 2)]     # (256, 1) is the primary


def s1_scales(smax):
    return np.unique(np.round(np.geomspace(8, smax, 12)).astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--np-root", default="D:/ieeg07/np")
    ap.add_argument("--data-root", default="D:/ieeg07/braintreebank")
    ap.add_argument("--index", default=os.path.join(HERE, "results", "index", "neuroprobe_lite_index.csv.gz"))
    ap.add_argument("--root", default="D:/ieeg07")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=None, help="testing only")
    ap.add_argument("--root-out", default=None, help="testing only: write outputs under this root")
    a = ap.parse_args()

    torch, laplacian, stft, Subject, LITE = neuroprobe_imports(a.np_root, a.data_root)
    tag = f"s{a.subject}_t{a.trial}"
    subj = Subject(a.subject, allow_corrupted=False, cache=False)
    labels = [e for e in LITE[subj.subject_identifier] if e in subj.electrode_labels]
    keys = [subj.h5_neural_data_keys[e] for e in labels]
    wf = session_windows(a.index, a.subject, a.trial, ["within"])
    if a.max_windows:
        wf = wf[: a.max_windows]
    if a.root_out:
        a.root = a.root_out
    nw, ne = len(wf), len(labels)
    print(f"{tag}: {nw} within-session windows x {ne} electrodes", flush=True)

    scratch = os.path.join(a.root, "scratch")
    os.makedirs(scratch, exist_ok=True)
    win_path = os.path.join(scratch, f"{tag}_sens_windows.f32")
    win = np.memmap(win_path, dtype=np.float32, mode="w+", shape=(nw, ne, N))
    t0 = time.time()
    with h5py.File(os.path.join(a.data_root, f"sub_{a.subject}_trial{a.trial:03}.h5"), "r") as f:
        idx = wf[:, None] + np.arange(N)[None, :]
        for j, k in enumerate(keys):
            win[:, j, :] = f["data"][k][:].astype(np.float32)[idx]
    win.flush()
    print(f"  read in {time.time() - t0:.0f} s", flush=True)

    s1_dirs = {st: os.path.join(a.root, f"features_S1_s{st[0]}_o{st[1]}") for st in S1_SETTINGS}
    s3_dir = os.path.join(a.root, "features_S3_raw")
    for d in list(s1_dirs.values()) + [s3_dir]:
        os.makedirs(d, exist_ok=True)
    S1 = {st: np.full((nw, ne, len(Q_ALL)), np.nan) for st in S1_SETTINGS}
    L1r = np.empty((nw, ne))
    HQr = np.empty((nw, ne, len(Q_ALL)))
    FLr = np.empty((nw, ne), dtype=np.int32)
    L5r = np.lib.format.open_memmap(os.path.join(s3_dir, f"{tag}_L5.npy"), mode="w+",
                                    dtype=np.float32, shape=(nw, ne, 17, 38))
    t1 = time.time()
    for b0 in range(0, nw, a.batch):
        b1 = min(nw, b0 + a.batch)
        x = torch.from_numpy(np.array(win[b0:b1], dtype=np.float32, copy=True))
        lap, _, _ = laplacian(x, labels, remove_non_laplacian=False)
        lap64 = lap.numpy().astype(np.float64).reshape(-1, N)
        for st in S1_SETTINGS:
            h = batch_mfdfa(lap64, s1_scales(st[0]), q=Q_POS, order=st[1], backend="torch")["hq"]
            block = S1[st][b0:b1]                      # a view: assigning into it writes S1[st]
            block[:, :, POS_COLS] = h.reshape(b1 - b0, ne, len(Q_POS))
        raw64 = x.numpy().astype(np.float64).reshape(-1, N)
        # A raw electrode can be constant for a whole second while its Laplacian is not (the Laplacian
        # subtracts the neighbours). Log variance and exponents are undefined there and are written as
        # NaN; the flat-run file marks them (run == N) and decode.py --exclude-constant drops those
        # windows, counted (PREREGISTRATION.md D5). decode.py still refuses any NaN that reaches it.
        var = raw64.var(axis=1)
        const = var <= 0
        with np.errstate(divide="ignore"):
            L1r[b0:b1] = np.where(const, np.nan, np.log(var)).reshape(b1 - b0, ne)
        hq = batch_mfdfa(raw64, SCALES, q=Q_ALL, backend="torch")["hq"]
        hq[const] = np.nan
        HQr[b0:b1] = hq.reshape(b1 - b0, ne, len(Q_ALL))
        FLr[b0:b1] = longest_flat_run(raw64).reshape(b1 - b0, ne)
        if not (FLr[b0:b1].reshape(-1)[const] == N).all():
            raise AssertionError("a zero-variance window whose flat run is not the whole window")
        L5r[b0:b1] = stft(x, preprocess="stft_abs", preprocess_parameters=STFT).numpy()
        if (b0 // a.batch) % 40 == 0:
            print(f"  {b1}/{nw} ({time.time() - t1:.0f} s)", flush=True)
    L5r.flush()
    L5r._mmap.close()
    win._mmap.close()
    del L5r, win, x, lap
    os.remove(win_path)

    for st in S1_SETTINGS:
        if not np.isfinite(S1[st][:, :, POS_COLS]).all():
            raise ValueError(f"S1 {st}: positive-q exponents not fully written")
    meta = dict(subject=a.subject, trial=a.trial, n_windows=nw, n_electrodes=ne, electrode_labels=labels,
                splits=["within"], seconds=round(time.time() - t0, 1))
    for st, d in s1_dirs.items():
        np.save(os.path.join(d, f"{tag}_window_from.npy"), wf)
        np.save(os.path.join(d, f"{tag}_hq.npy"), S1[st])
        json.dump(dict(meta, analysis="S1", s_max=st[0], order=st[1], scales=s1_scales(st[0]).tolist()),
                  open(os.path.join(d, f"{tag}_meta.json"), "w"), indent=1)
    np.save(os.path.join(s3_dir, f"{tag}_window_from.npy"), wf)
    np.save(os.path.join(s3_dir, f"{tag}_L1.npy"), L1r)
    np.save(os.path.join(s3_dir, f"{tag}_hq.npy"), HQr)
    np.save(os.path.join(s3_dir, f"{tag}_flat.npy"), FLr)
    json.dump(dict(meta, analysis="S3", reference="raw", scales=SCALES.tolist()),
              open(os.path.join(s3_dir, f"{tag}_meta.json"), "w"), indent=1)
    print(f"{tag}: done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
