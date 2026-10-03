# The study's pre-registration

These are the committed versions of the study's pre-registration, oldest first. Version 1 is the
registration as fixed before any feature set was decoded; each later version adds dated deviations.

**Not deposited in a public registry.** Each version was committed to the author's own git repository;
the commit times below are recorded by that repository, not by a third party. The first decoding of a
ladder feature set produced `results/decode/L0.csv`, created 2026-10-01 09:46:52 -0500
(a filesystem time). The only earlier computation on the recordings reproduced Neuroprobe's published
baseline on one cell, as version 1's header states.

**One clause is redacted** in every version: a sentence in deviation D0 that refers to an unpublished
analysis of other data, replaced by a marked placeholder. Nothing else differs from the committed text.
`provenance.json` gives, for each version, the SHA-256 of the unredacted original and of this copy, so
anyone given the original can verify both.

| Version | Commit | Committed | Commit message | SHA-256 of the original |
|---|---|---|---|---|
| v1 | `4c121a8e21bd` | 2026-10-01 07:26:52 -0500 | Pre-registration: amplitude-invariance ladder, fixed before any decodi | `770492d0398673d0...` |
| v2 | `75699dd398b2` | 2026-10-01 07:31:56 -0500 | D1 (circular electrode selection) added before running; gates F2/F3 pa | `a8faf2e683cf7c5a...` |
| v3 | `cd4187196098` | 2026-10-01 07:37:28 -0500 | Gate 3 passed (accuracy identical, AUROC within 1e-4); D2: PopT per-wi | `8865e8da293a2a47...` |
| v4 | `188365c856d5` | 2026-10-01 09:48:38 -0500 | D3 (POST HOC): within-session fold construction may leak time; logged  | `7028aca737a74bf1...` |
| v5 | `394a77e3cf2f` | 2026-10-01 11:26:55 -0500 | D4 (POST HOC): published linear baseline stops after 4 lbfgs iteration | `ea82bf88b249eafe...` |
| v6 | `478cfd808110` | 2026-10-01 21:11:54 -0500 | D5: constant raw-voltage windows in S3 (6 of 134,087 windows x electro | `311f135e8724e1a2...` |
