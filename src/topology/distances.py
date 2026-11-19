"""
Bottleneck distance between persistence diagrams.

Bottleneck distance is the primary statistical comparison metric in this
experiment. The Saturday probe validated that bottleneck distance reliably
discriminates topology types where exact Betti number recovery is brittle:
on canonical shapes embedded in R^768, bottleneck distance between
different-topology shapes was 11-20x larger than between same-topology
samples of different points.

Per Section 4.2 of the technical paper:

  The primary test for each arm is whether the bottleneck distance between
  contradiction and matched-baseline persistence diagrams differs significantly
  from the bottleneck distance between two matched-baseline instances (the
  null reference) and whether this difference concentrates at a specific
  dimension.
"""

from __future__ import annotations

import numpy as np
from persim import bottleneck


def bottleneck_distance(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    """
    Bottleneck distance between two persistence diagrams in a single dimension.

    Filters out infinite-death (essential) features before computation,
    since persim's bottleneck implementation requires finite values.
    Returns 0 if both diagrams are empty after filtering.
    """
    if len(dgm_a) == 0 and len(dgm_b) == 0:
        return 0.0
    a_finite = dgm_a[np.isfinite(dgm_a[:, 1])] if len(dgm_a) else np.array([]).reshape(0, 2)
    b_finite = dgm_b[np.isfinite(dgm_b[:, 1])] if len(dgm_b) else np.array([]).reshape(0, 2)
    if len(a_finite) == 0 and len(b_finite) == 0:
        return 0.0
    return float(bottleneck(a_finite, b_finite))


def pairwise_bottleneck_matrix(diagrams: list[np.ndarray]) -> np.ndarray:
    """Symmetric matrix of pairwise bottleneck distances. Diagonal is zero."""
    n = len(diagrams)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = bottleneck_distance(diagrams[i], diagrams[j])
            D[i, j] = d
            D[j, i] = d
    return D
