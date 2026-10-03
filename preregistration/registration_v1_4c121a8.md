# Pre-Registration: what do intracranial decoders read out of a one-second window?

**Fixed:** 2026-10-01, before any feature set defined below had been decoded on Brain Treebank data.
**Provenance:** committed to this project's local git repository before `src/` contained any
decoding code for these feature sets. The only real-data computation that preceded this file is the
reproduction of Neuroprobe's own published linear baseline (gate 3, section 9), which re-derives a
number already on the public leaderboard and tests the pipeline, not a hypothesis.

> The killer objection to a paper like this is *"you chose the features, the scales and the
> comparison after seeing which ones won."* This document is the defence. Every feature set, every
> estimator setting, the classifier, the statistics and the direction of every prediction are fixed
> here. Anything done differently later is logged in section 10 with a date and a reason.

---

## 1. Why the brief's question was replaced, before any data

The project brief (CLAUDE.md section 1) proposed that a nonlinear-dynamics battery would win on
"slow, state-like" tasks (arousal, engagement) and lose on "fast evoked" ones. Reading the Neuroprobe
paper and code (2026-10-01) showed that this cannot be tested on Neuroprobe as written:

1. **Every Neuroprobe task is a one-second window starting at a word onset** (paper, Approach;
   `config.py` START/END = 0/1 s). There are no arousal or engagement tasks.
2. **Amplitude is the confound the paper itself names.** Supplementary Figure 16 attributes part of
   the gap between the linear baseline and BrainBERT/PopT to normalisation: the linear baseline
   z-scores across the training set, while BrainBERT and PopT z-score each one-second window, which
   discards the window's absolute amplitude.
3. **DFA and MFDFA exponents are invariant to per-window affine rescaling** (project 04: `normalise`
   is an exact null; `tests/test_batch_dfa.py::test_amplitude_invariance` here). A dynamics battery is
   therefore amplitude-free by construction, exactly like the per-window-normalised foundation models.

So the sharper, testable question is not "dynamics or deep nets" but:

> **On Neuroprobe Lite, how much of what decoders read out of a one-second intracranial window is the
> window's absolute amplitude, and how much is amplitude-invariant structure? And once amplitude is
> removed from both sides, does a small, validated dynamics battery match the foundation models that
> also remove it?**

The timescale idea survives in a form the data can test (H4): tasks differ in how slowly their
*labels* change across the movie, and that is measurable from annotations alone.

## 2. Data

* **Benchmark:** Neuroprobe Lite as shipped at commit `b901984` (2026-09-15): 12 sessions
  (subject, trial) = (1,1) (1,2) (2,0) (2,4) (3,0) (3,1) (4,0) (4,1) (7,0) (7,1) (10,0) (10,1); at most
  120 electrodes per subject from `NEUROPROBE_LITE_ELECTRODES`; at most 3,500 class-balanced samples
  per task; 15 binary tasks; three split types.
* **Windows and labels are Neuroprobe's, not mine.** They are obtained from
  `BrainTreebankSubjectTrialBenchmarkDataset(..., output_indices=True)` and the three
  `generate_splits_*` functions, so every window boundary, label and fold assignment is the
  benchmark's. Test sets are exactly what `examples/eval_population.py` scores: the second half of
  each held-out fold (the first half, `val_dataset`, is unused by the linear pipeline and stays
  unused here).
* **Raw data:** Brain Treebank, hashed at download (`MANIFEST.jsonl`; no published checksums exist).
* **Electrodes:** the Lite list, with Neuroprobe's own corrupted-electrode exclusion
  (`allow_corrupted=False`) left in place.

## 3. Preprocessing, identical for every feature set

* Laplacian re-referencing by Neuroprobe's `laplacian_rereference_neural_data`, with the
  `remove_non_laplacian=False` setting their `laplacian` preprocessing option uses.
* No filtering and no line-noise removal (the published linear baseline uses neither).
* Window: samples `[onset, onset + 2048)` at 2048 Hz.

## 4. The ladder: six feature sets that differ in one property

| Rung | Features per electrode | Amplitude retained? | Neural data? |
|---|---|---|---|
| **L0** time only | window start time in the movie (seconds), one feature per window | n/a | **no** |
| **L1** amplitude only | log variance of the window (1) | **only** amplitude | yes |
| **L2** dynamics only | h(q) at q = 1, 2, 3, 5 (4) | **no** (invariant) | yes |
| **L3** dynamics + amplitude | L1 and L2 together (5) | yes | yes |
| **L4** spectral shape only | Neuroprobe's STFT magnitude divided by its own sum over time and frequency, per electrode per window | **no** | yes |
| **L5** published baseline | Neuroprobe's Laplacian STFT magnitude (`nperseg` 512, 75% overlap, Hann, 0-150 Hz) | yes | yes |

**Foundation models** are taken from the leaderboard JSONs at the same commit, which report test
AUROC per session per fold on identical splits: BrainBERT (frozen, off-the-shelf, per-window STFT
z-scoring), PopulationTransformer (off-the-shelf, per-window STFT z-scoring), DIVER-1 (0.1 s tiny,
frozen and fine-tuned), MAPA. No foundation model is re-run for the primary analysis.

### The dynamics battery, frozen

* Estimator: `src/batch_dfa.py`, validated against fdnkit at commit `c575043` to 1e-9 on clean
  signals (`tests/test_batch_dfa.py`, 17 tests).
* Detrending order 1. Scales: 12 log-spaced integers from 8 to N/8 = 256 samples, the same rule used
  by the known-truth arm. Scale-relative floor 1e-3 (fdnkit's default).
* **Positive q only in the primary battery**, q = 1, 2, 3, 5 (h(2) is the DFA exponent). Negative-q
  moments on a 2048-sample window are dominated by any constant run, and the value is then set by
  floating-point rounding rather than by the signal (section 3 of `tests/test_batch_dfa.py`): a
  single plateau inflates the multifractal width of a monofractal past 2. They enter only the
  secondary analysis S2, on QC-passed windows.
* Why these choices and not others: order 1 and an N/8 upper scale are the conventional defaults
  that project 04's pre-registered grid found stable; the sensitivity grid S1 reports the
  alternatives so no single choice carries the result.

## 5. Classifier, identical for every rung

Neuroprobe's linear pipeline, unchanged: features flattened per window, `StandardScaler` fit on the
training set, `LogisticRegression(random_state=42, max_iter=10000, tol=1e-3)`, test AUROC. For the
cross-subject split, per-electrode features are averaged within Desikan-Killiany regions present in
both subjects using Neuroprobe's own `combine_regions`, exactly as the published baseline does.

## 6. Outcomes

* **Cell:** test AUROC for one (split, session, task), averaged over that cell's folds (2 folds
  within-session; 1 otherwise), as the leaderboard does.
* **Overall:** the mean over the 15 tasks, the leaderboard's "overall".
* **Inference unit:** the session. Subjects contribute two sessions each, so uncertainty comes from
  a **hierarchical bootstrap** -- resample subjects with replacement, then sessions within each drawn
  subject -- 10,000 resamples, seed fixed as a pure function of the comparison (never `hash()`).
* **Differences, not ratios.** Every comparison is a paired difference in AUROC with its bootstrap
  95% interval. No ratio of AUROC excesses is reported anywhere: tasks whose baseline sits near 0.5
  make such ratios divide by almost nothing (project 08, C-series).

## 7. Hypotheses, with direction, fixed in advance

| | Statement | Test | Prediction |
|---|---|---|---|
| **H1** | Removing amplitude costs decoding performance | within-session overall, L5 minus L4 | **positive**, interval excludes 0 |
| **H2** | Amplitude alone decodes far above chance | within-session overall, L1 | **above 0.55** |
| **H3** | With amplitude removed from both sides, the dynamics battery matches per-window-normalised foundation models | L2 minus BrainBERT and L2 minus PopT (per-window variants), within-session overall | **equivalent**: interval inside +/-0.02 AUROC |
| **H4** | The dynamics battery does relatively better on tasks whose labels change slowly | Spearman rho across the 15 tasks between label timescale tau (below) and L2 minus L4, within-session | **rho > 0** |
| **H5** | Time in the movie alone does not decode the labels | L0 per task, within-session | **interval includes 0.5** for each task; any task where it does not is reported as confounded |

**The equivalence margin of 0.02 is fixed here**, before seeing any L2 number, and is chosen from the
public leaderboard: it is about the size of DIVER-1's margin over the next model and smaller than the
0.035 gap between the top submission and the linear baseline within-session.

**Label timescale tau, defined before computing it.** For each session and task, place the task's
underlying annotation on a 1-second grid over the movie (the mean of the variable over words
starting in that second, or for indicator tasks the fraction of those words in the positive class;
for Speech, whether any word starts in that second). tau is the first lag, in seconds, at which the
NaN-aware autocorrelation falls below 1/e, capped at 600 s. A task's tau is its median over the 12
sessions. This uses annotations only, never neural data.

**H3 is the brief's thesis in testable form. It can fail in either direction**, and both outcomes are
reported: dynamics clearly worse means the foundation models extract amplitude-invariant structure a
four-exponent battery does not capture; clearly better means the foundation models lose something the
battery keeps.

## 8. Secondary and sensitivity analyses, also fixed now

* **S1, estimator settings.** L2 recomputed over upper scale {N/16, N/8, N/4} x order {1, 2}; all six
  reported, primary marked.
* **S2, negative q.** L2 extended with q = -3, -2, -1, 0 on windows where every electrode's longest
  constant run is below the smallest scale (8 samples); excluded windows are counted per session.
  This analysis also reports how common constant runs are in Brain Treebank, which project NB found
  ranges from 1.2% to 30% of channels across EEG corpora.
* **S3, input.** L1, L2 and L5 on raw voltage instead of the Laplacian.
* **S4, other splits.** Every rung on the cross-session and cross-subject splits. H1-H5 are stated
  for within-session because it has the most data; the other splits are reported in full but are not
  separate confirmatory tests.

## 9. Gates that precede the analysis, and what stops it

| Gate | Criterion | Status |
|---|---|---|
| Estimability | At N = 2048 the battery recovers known exponents on fGn and fBm (`src/known_truth.py`) | run 2026-10-01; reported with the results |
| Cost | The battery is tractable on this machine | **passed**: 0.4 ms per window batched, about 50x faster than the reference |
| Reproduction | Neuroprobe's own script, run here, reproduces the published Laplacian+STFT AUROC for sub 1 trial 1 Sentence Onset within-session to within 0.01 per fold (published 0.9556, 0.8907) | in progress |

**If the reproduction gate fails, nothing in section 7 is run** until the discrepancy is explained,
and the discrepancy is itself reported.

## 10. Deviations

None yet. Each deviation is added here with its date, what changed, why, and whether it was decided
before or after seeing the outcome it affects.

### D0: 2026-10-01: departures from the project brief, decided before any data
* The state-vs-evoked dissociation is replaced by the amplitude ladder and H4 (section 1).
* Recurrence quantification is dropped from the battery: O(N^2) per window across about 10^7
  windows, and embedding dimension, delay and threshold are three free parameters each of which
  would need its own justification.
* Fractional-order network (FODN) descriptors are dropped from the primary battery. They are
  multivariate across 120 electrodes per window, and [REDACTED: one clause referring to an unpublished analysis of other data]. They may enter later only as a logged deviation.
* The electrode-identity negative control in the brief is deferred: an intracranial paper
  (arXiv:2510.27090) shows encoders learn a channel's functional identity *on purpose*, to
  aggregate across subjects, so in iEEG decodable identity is not by itself evidence of a shortcut.
