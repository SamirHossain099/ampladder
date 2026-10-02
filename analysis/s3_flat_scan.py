"""Count constant (zero-variance) raw-voltage electrode-windows in every Lite session.

S3 evaluates L1, L2 and L5 on raw voltage. The primary analysis found no zero-variance window on the
Laplacian input (FINDINGS.md F13), but a raw electrode can be flat while its Laplacian is not, because
the Laplacian subtracts the neighbours. S3 extraction stopped on such a window on 2026-10-01
(s2_t0). This scan measures the scope before any handling is chosen, so the choice is made against
the count and not against a result. Read-only; writes results/s3/raw_flat_windows.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from extract_features import neuroprobe_imports, session_windows  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2048
SESSIONS = [(1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1), (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1)]


def main(np_root="D:/ieeg07/np", data_root="D:/ieeg07/braintreebank"):
    _, _, _, Subject, LITE = neuroprobe_imports(np_root, data_root)
    index = os.path.join(HERE, "results", "index", "neuroprobe_lite_index.csv.gz")
    rows = []
    for s, t in SESSIONS:
        subj = Subject(s, allow_corrupted=False, cache=False)
        labels = [e for e in LITE[subj.subject_identifier] if e in subj.electrode_labels]
        wf = session_windows(index, s, t, ["within"])
        with h5py.File(os.path.join(data_root, f"sub_{s}_trial{t:03}.h5"), "r") as f:
            for e in labels:
                x = f["data"][subj.h5_neural_data_keys[e]][:].astype(np.float32)
                w = np.lib.stride_tricks.sliding_window_view(x, N)[wf]
                flat = (w.max(axis=1) == w.min(axis=1))
                if flat.any():
                    rows.append(dict(subject=s, trial=t, electrode=e, n_windows=len(wf), n_flat=int(flat.sum())))
        tot = sum(r["n_flat"] for r in rows if (r["subject"], r["trial"]) == (s, t))
        print(f"s{s}_t{t}: {len(labels)} electrodes x {len(wf)} windows, {tot} flat electrode-windows", flush=True)
    df = pd.DataFrame(rows, columns=["subject", "trial", "electrode", "n_windows", "n_flat"])
    os.makedirs(os.path.join(HERE, "results", "s3"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "results", "s3", "raw_flat_windows.csv"), index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
