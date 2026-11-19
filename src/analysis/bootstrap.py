"""
Paired bootstrap of the max-persistence dimensional statistic.

This module implements the primary effect-size machinery preregistered in
docs/analysis_protocol.md (Sections 4 and 5). It supersedes the earlier
within/across-group bottleneck-distance formulation.

Statistic.
  Per item, per dimension k, the scalar is max persistence: the lifetime of
  the single most persistent finite H_k feature (Section 4.1). Infinite
  (essential) features are ignored, since an essential class has no finite
  lifetime to compare. An optional per-item scale divides the lifetimes
  (Section 4.2: per-token arm divides by point-cloud diameter; attention arm
  is already on a [0, 1] filtration and uses scale 1.0).

  The unit of analysis is the matched pair (contradiction item, coherent-edit
  control item). For pair p the signed contrast is
      Delta_k(p) = M_k(contradiction_p) - M_k(coherent_edit_p).
  The effect size is the paired (one-sample) Cohen's d_z over pairs:
      d_z(k) = mean_p[Delta_k(p)] / sd_p[Delta_k(p)]   (ddof=1).

Resampling.
  The bootstrap resamples the unit with replacement. By default the unit is
  the pair. When a unit_ids array is supplied (source-family ids per Section
  2), whole families are resampled together so that multiple variants from
  one source are not treated as independent observations.

Bottleneck distance is intentionally not here anymore. It lives in
src/topology/distances.py and is used for figures and secondary diagnostics,
not for the primary statistic.
"""

from __future__ import annotations

import numpy as np


def max_persistence(diagram: np.ndarray | None, scale: float = 1.0) -> float:
    """Lifetime of the most persistent finite feature in a persistence diagram.

    Args:
        diagram: shape (n_features, 2) with columns [birth, death]. Death may
            contain np.inf for essential classes. None or empty returns 0.0.
        scale: positive divisor applied to lifetimes (e.g. point-cloud
            diameter on the per-token arm). Must be > 0.

    Returns:
        max over finite features of (death - birth) / scale, or 0.0 if there
        are no finite features.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if diagram is None:
        return 0.0
    arr = np.asarray(diagram, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] == 0:
        return 0.0
    births = arr[:, 0]
    deaths = arr[:, 1]
    finite = np.isfinite(deaths)
    if not finite.any():
        return 0.0
    lifetimes = deaths[finite] - births[finite]
    return float(lifetimes.max()) / float(scale)


def paired_cohens_dz(
    contradiction: np.ndarray,
    coherent_edit: np.ndarray,
) -> float:
    """Paired (one-sample) Cohen's d_z over matched pairs.

    Args:
        contradiction: 1D array of M_k scalars, one per pair (the
            contradiction member).
        coherent_edit: 1D array of M_k scalars, same length and pair-aligned
            (the coherent-edit control member).

    Returns:
        mean(delta) / sd(delta, ddof=1), where delta = contradiction -
        coherent_edit. Returns 0.0 if fewer than 2 pairs or the standard
        deviation of the differences is 0.
    """
    c = np.asarray(contradiction, dtype=np.float64).ravel()
    e = np.asarray(coherent_edit, dtype=np.float64).ravel()
    if c.shape != e.shape:
        raise ValueError(
            f"contradiction and coherent_edit must align; got {c.shape} and {e.shape}"
        )
    if c.size < 2:
        return 0.0
    delta = c - e
    sd = float(np.std(delta, ddof=1))
    if sd <= 0:
        return 0.0
    return float(np.mean(delta)) / sd


def _resample_units(
    n_pairs: int,
    unit_ids: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an array of pair indices drawn by resampling units with replacement.

    If unit_ids is None each pair is its own unit. Otherwise pairs sharing a
    unit id are resampled together (source-family resampling).
    """
    if unit_ids is None:
        return rng.integers(0, n_pairs, size=n_pairs)
    unit_ids = np.asarray(unit_ids)
    if unit_ids.shape[0] != n_pairs:
        raise ValueError(
            f"unit_ids length {unit_ids.shape[0]} != n_pairs {n_pairs}"
        )
    uniq = np.unique(unit_ids)
    # Map each unit to its member pair indices.
    members = {u: np.where(unit_ids == u)[0] for u in uniq}
    chosen = rng.integers(0, uniq.shape[0], size=uniq.shape[0])
    picked = [members[uniq[i]] for i in chosen]
    return np.concatenate(picked) if picked else np.empty(0, dtype=int)


def paired_bootstrap(
    contradiction: np.ndarray,
    coherent_edit: np.ndarray,
    unit_ids: np.ndarray | None = None,
    n_iterations: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap distribution of the paired Cohen's d_z.

    Resamples the unit (pair, or source-family if unit_ids given) with
    replacement n_iterations times and recomputes d_z each time.

    Args:
        contradiction, coherent_edit: pair-aligned 1D arrays of M_k scalars.
        unit_ids: optional per-pair source-family id for family-level
            resampling. None means each pair is its own unit.
        n_iterations: number of bootstrap iterations.
        seed: RNG seed.

    Returns:
        np.ndarray of shape (n_iterations,), one d_z per iteration.

    Raises:
        ValueError: if inputs misalign, fewer than 2 pairs, or n_iterations <= 0.
    """
    c = np.asarray(contradiction, dtype=np.float64).ravel()
    e = np.asarray(coherent_edit, dtype=np.float64).ravel()
    if c.shape != e.shape:
        raise ValueError(
            f"contradiction and coherent_edit must align; got {c.shape} and {e.shape}"
        )
    if c.size < 2:
        raise ValueError(f"need at least 2 pairs, got {c.size}")
    if n_iterations <= 0:
        raise ValueError(f"n_iterations must be > 0, got {n_iterations}")

    n_pairs = c.size
    rng = np.random.default_rng(seed)
    out = np.zeros(n_iterations, dtype=np.float64)
    for it in range(n_iterations):
        idx = _resample_units(n_pairs, unit_ids, rng)
        out[it] = paired_cohens_dz(c[idx], e[idx])
    return out


def bootstrap_summary(
    distribution: np.ndarray,
    confidence_level: float = 0.95,
) -> dict:
    """Summarize a bootstrap distribution.

    Returns:
        dict with keys: point_estimate (mean), median, lower_ci, upper_ci, std.
    """
    alpha = 1.0 - confidence_level
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    return {
        "point_estimate": float(np.mean(distribution)),
        "median": float(np.median(distribution)),
        "lower_ci": float(np.percentile(distribution, lower_q)),
        "upper_ci": float(np.percentile(distribution, upper_q)),
        "std": float(np.std(distribution, ddof=1)),
    }


# ----- Sanity check (run: python -m src.analysis.bootstrap) -----


def _sanity_check() -> None:
    """Hand-derivable cases."""
    import sys

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        marker = "PASS" if cond else "FAIL"
        tail = f": {detail}" if (detail and not cond) else ""
        print(f"  [{marker}] {name}{tail}")
        if not cond:
            failures.append(name)

    # --- max_persistence ---

    # Case 1: empty diagram -> 0.
    check("max_persistence of empty is 0", max_persistence(np.empty((0, 2))) == 0.0)
    check("max_persistence of None is 0", max_persistence(None) == 0.0)

    # Case 2: [[0,1],[0.5,2]] -> max(1.0, 1.5) = 1.5.
    dgm = np.array([[0.0, 1.0], [0.5, 2.0]])
    check(
        "max_persistence picks longest lifetime (1.5)",
        abs(max_persistence(dgm) - 1.5) < 1e-12,
        detail=f"got {max_persistence(dgm)}",
    )

    # Case 3: scale divides. Same diagram, scale 3 -> 0.5.
    check(
        "max_persistence divides by scale (1.5/3 = 0.5)",
        abs(max_persistence(dgm, scale=3.0) - 0.5) < 1e-12,
        detail=f"got {max_persistence(dgm, scale=3.0)}",
    )

    # Case 4: infinite death ignored. [[0,inf],[0,2]] -> 2.0.
    dgm_inf = np.array([[0.0, np.inf], [0.0, 2.0]])
    check(
        "max_persistence ignores infinite features (-> 2.0)",
        abs(max_persistence(dgm_inf) - 2.0) < 1e-12,
        detail=f"got {max_persistence(dgm_inf)}",
    )

    # Case 5: all-infinite -> 0 (no finite feature).
    check(
        "max_persistence of all-infinite is 0",
        max_persistence(np.array([[0.0, np.inf]])) == 0.0,
    )

    # Case 6: scale <= 0 rejected.
    try:
        max_persistence(dgm, scale=0.0)
        check("scale=0 rejected", False)
    except ValueError:
        check("scale=0 rejected with ValueError", True)

    # --- paired_cohens_dz ---

    # Case 7: zero differences -> 0.
    check(
        "d_z of identical members is 0",
        paired_cohens_dz(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0,
    )

    # Case 8: hand-derived value.
    # contra = [3,1,4,2], edit = [1,1,1,1]; delta = [2,0,3,1].
    # mean = 1.5; var(ddof=1) = (0.25+2.25+2.25+0.25)/3 = 5/3; sd = 1.290994.
    # d_z = 1.5 / 1.290994 = 1.161895.
    dz = paired_cohens_dz(np.array([3.0, 1.0, 4.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0]))
    check(
        "d_z hand case ~= 1.161895",
        abs(dz - 1.161895) < 1e-5,
        detail=f"got {dz}",
    )

    # Case 9: sign flips when members swap.
    dz_pos = paired_cohens_dz(np.array([3.0, 4.0, 5.0]), np.array([1.0, 1.0, 1.0]))
    dz_neg = paired_cohens_dz(np.array([1.0, 1.0, 1.0]), np.array([3.0, 4.0, 5.0]))
    check("d_z sign convention: contra>edit positive", dz_pos > 0, detail=f"got {dz_pos}")
    check("d_z sign convention: contra<edit negative", abs(dz_neg + dz_pos) < 1e-12)

    # Case 10: fewer than 2 pairs -> 0.
    check("d_z with one pair is 0", paired_cohens_dz(np.array([5.0]), np.array([1.0])) == 0.0)

    # Case 11: misaligned inputs rejected.
    try:
        paired_cohens_dz(np.array([1.0, 2.0]), np.array([1.0]))
        check("misaligned d_z rejected", False)
    except ValueError:
        check("misaligned d_z rejected with ValueError", True)

    # --- paired_bootstrap ---

    rng = np.random.default_rng(0)
    n = 30

    # Case 12: separated -> mean d_z large, lower CI > 0.
    contra = 5.0 + rng.normal(0, 0.3, n)
    edit = 1.0 + rng.normal(0, 0.3, n)
    dist = paired_bootstrap(contra, edit, n_iterations=500, seed=7)
    summ = bootstrap_summary(dist)
    check(
        "bootstrap separated: mean d_z > 2",
        summ["point_estimate"] > 2.0,
        detail=f"got {summ['point_estimate']:.3f}",
    )
    check(
        "bootstrap separated: lower CI > 0",
        summ["lower_ci"] > 0,
        detail=f"got {summ['lower_ci']:.3f}",
    )

    # Case 13: no difference -> |mean d_z| small.
    contra2 = 2.0 + rng.normal(0, 0.3, n)
    edit2 = 2.0 + rng.normal(0, 0.3, n)
    dist2 = paired_bootstrap(contra2, edit2, n_iterations=500, seed=7)
    summ2 = bootstrap_summary(dist2)
    check(
        "bootstrap no-difference: |mean d_z| < 0.5",
        abs(summ2["point_estimate"]) < 0.5,
        detail=f"got {summ2['point_estimate']:.3f}",
    )

    # Case 14: reproducible under seed.
    da = paired_bootstrap(contra, edit, n_iterations=100, seed=42)
    db = paired_bootstrap(contra, edit, n_iterations=100, seed=42)
    check("same seed gives same distribution", np.array_equal(da, db))

    # Case 15: summary keys.
    check(
        "bootstrap_summary has expected keys",
        set(summ.keys()) == {"point_estimate", "median", "lower_ci", "upper_ci", "std"},
    )

    # Case 16: source-family resampling. 4 pairs in 2 families of 2; every
    # resample draws whole families, so the resampled index count is even and
    # a multiple of the family size.
    c4 = np.array([5.0, 5.1, 1.0, 1.1])
    e4 = np.array([1.0, 1.0, 1.0, 1.0])
    fam = np.array([0, 0, 1, 1])
    rng_f = np.random.default_rng(3)
    sampled = _resample_units(4, fam, rng_f)
    check(
        "family resample returns whole families (len multiple of 2)",
        sampled.shape[0] % 2 == 0 and sampled.shape[0] == 4,
        detail=f"got len {sampled.shape[0]}",
    )
    dist_f = paired_bootstrap(c4, e4, unit_ids=fam, n_iterations=100, seed=5)
    check("family bootstrap runs and returns 100 values", dist_f.shape[0] == 100)

    # Case 17: input validation.
    try:
        paired_bootstrap(np.array([1.0]), np.array([1.0]), n_iterations=10)
        check("bootstrap with <2 pairs rejected", False)
    except ValueError:
        check("bootstrap with <2 pairs rejected with ValueError", True)
    try:
        paired_bootstrap(contra, edit, n_iterations=0)
        check("bootstrap n_iterations=0 rejected", False)
    except ValueError:
        check("bootstrap n_iterations=0 rejected with ValueError", True)

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {failures}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    print("bootstrap sanity check\n")
    _sanity_check()
