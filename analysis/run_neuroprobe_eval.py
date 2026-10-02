"""Run Neuroprobe's own `examples/eval_population.py`, unmodified, with one runtime patch.

Gate 3 of the brief: reproduce a published leaderboard number *before adding anything*. The
reproduction has to go through Neuroprobe's code path, or a mismatch could be mine rather than
theirs, and a match could be a coincidence of two pipelines that happen to agree.

The one patch: `eval_population.py` constructs `BrainTreebankSubject(subject_id, cache=True)`,
which holds the whole session in RAM -- about 10 GB for 120 electrodes x 21.4 M samples x float32
on sub 1 / trial 1. This machine has 31.6 GB with roughly 9.5 GB free while other work runs. With
`cache=False` the subject reads each window from the HDF5 file instead. That changes where the
bytes come from, not which bytes or what is done with them, so the computation is unchanged.

Everything else -- the dataset, the splits, Laplacian re-referencing, the STFT, StandardScaler,
LogisticRegression(max_iter=10000, tol=1e-3), the AUROC -- is Neuroprobe's code, run via runpy.

Usage (inside the project venv):
    python src/run_neuroprobe_eval.py --np-root D:/ieeg07/np --data-root D:/ieeg07/braintreebank \
        -- --subject_id 1 --trial_id 1 --eval_name onset --split_type WithinSession --only_1second \
           --save_dir results/neuroprobe_repro
"""
import argparse
import os
import runpy
import sys


def main():
    if "--" not in sys.argv:
        raise SystemExit("separate this script's flags from eval_population.py's with --")
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--np-root", required=True)
    ap.add_argument("--data-root", required=True)
    a = ap.parse_args(sys.argv[1:cut])
    passthrough = sys.argv[cut + 1:]

    os.environ["ROOT_DIR_BRAINTREEBANK"] = a.data_root
    sys.path.insert(0, os.path.join(a.np_root, "examples"))
    sys.path.insert(0, a.np_root)

    import neuroprobe.braintreebank_subject as bts

    original_init = bts.BrainTreebankSubject.__init__

    def init_without_cache(self, *args, **kwargs):
        kwargs["cache"] = False
        original_init(self, *args, **kwargs)

    bts.BrainTreebankSubject.__init__ = init_without_cache

    script = os.path.join(a.np_root, "examples", "eval_population.py")
    sys.argv = [script] + passthrough
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
