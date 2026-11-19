"""
Vietoris-Rips persistent homology via ripser.

This is the primary topology computation for the sentence-embedding arm.
For each instance, we compute persistence diagrams up to dimension 4 on the
per-token contextual embedding point cloud, with sparse-filtration
approximation applied above a configurable point-cloud size for tractability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ripser import ripser


@dataclass
class PersistenceResult:
    """Output of a single persistent homology computation.

    diagrams[k] is the persistence diagram in dimension k as an array of
    shape (n_features, 2) with columns [birth, death]. Infinite-death
    features (essential classes) have np.inf in the death column.
    """
    diagrams: list[np.ndarray]
    max_dim: int
    n_points: int
    used_sparse: bool

    @property
    def n_features_per_dim(self) -> list[int]:
        """Number of finite-persistence features per dimension."""
        return [int(np.isfinite(d[:, 1]).sum()) for d in self.diagrams]


def compute_vr_persistence(point_cloud: np.ndarray, max_dim: int = 4,
                          sparse_threshold: int = 500) -> PersistenceResult:
    """
    Compute Vietoris-Rips persistence up to max_dim on a point cloud.

    If the point cloud is larger than sparse_threshold, ripser's sparse
    filtration approximation is used to keep computation tractable. The
    sparse approximation has theoretical guarantees on persistence-diagram
    distance from the exact result; see ripser documentation.

    Returns a PersistenceResult with the diagrams.
    """
    n = point_cloud.shape[0]
    use_sparse = n > sparse_threshold
    if use_sparse:
        # ripser's sparse mode: use a subset of edges via approximation
        # The 'thresh' parameter limits the filtration scale to control memory
        result = ripser(point_cloud, maxdim=max_dim, thresh=np.inf,
                       n_perm=sparse_threshold, coeff=2)  # Z/2Z, Section 10
    else:
        result = ripser(point_cloud, maxdim=max_dim, coeff=2)  # Z/2Z, Section 10
    return PersistenceResult(
        diagrams=result["dgms"],
        max_dim=max_dim,
        n_points=n,
        used_sparse=use_sparse,
    )


def bootstrap_persistence(point_cloud: np.ndarray,
                          n_bootstrap: int = 20,
                          subsample_size: int = 500,
                          max_dim: int = 4,
                          sparse_threshold: int = 500,
                          seed: int = 42) -> list[PersistenceResult]:
    """
    Bootstrap persistent homology via subsampling.

    For each bootstrap iteration, subsample n_subsample points without
    replacement and compute persistence. Returns the list of results,
    one per iteration, for downstream bootstrap confidence intervals.
    """
    rng = np.random.default_rng(seed)
    n = point_cloud.shape[0]
    sub_n = min(subsample_size, n)
    results = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=sub_n, replace=False)
        subsample = point_cloud[idx]
        result = compute_vr_persistence(subsample, max_dim=max_dim,
                                       sparse_threshold=sparse_threshold)
        results.append(result)
    return results
