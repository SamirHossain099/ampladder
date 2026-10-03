"""ampladder: checks for intracranial decoding benchmarks.

* mfdfa      batched DFA/MFDFA, validated against fdnkit, CPU and GPU backends
* ladder     the amplitude ladder's feature sets (amplitude only, dynamics, spectral shape)
* foldaudit  time leakage in a benchmark's folds, from its index alone
* selection  circular versus cross-fitted electrode selection (Neuroprobe's rule, ported verbatim)

"""
__version__ = "0.1.2"

from . import foldaudit, ladder, mfdfa, selection  # noqa: F401
