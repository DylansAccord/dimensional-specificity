"""
Matched-pair permutation test for the max-persistence dimensional statistic.

This is the inferential anchor preregistered in docs/analysis_protocol.md
(Section 4.4). It replaces the earlier bottleneck-distance permutation test.

Setup.
  Input is two pair-aligned arrays of max-persistence scalars at a fixed
  dimension k: the contradiction member and the coherent-edit control member
  of each matched pair. For pair p the signed contrast is
      Delta_k(p) = M_k(contradiction_p) - M_k(coherent_edit_p).

Test statistic.
  The observed statistic is the mean contrast, mean_p[Delta_k(p)].

Null and permutation.
  Under H0, the contradiction and coherent-edit labels are exchangeable
  within a pair. Swapping the two members of pair p negates Delta_k(p), so a
  within-pair label swap is exactly a sign flip of that pair's contrast. Each
  permutation draws an independent random sign per pair and recomputes the
  mean. The null distribution is therefore the distribution of mean(+/- Delta)
  over random sign assignments, which is symmetric about 0.

P-value.
  Two-sided, with the standard finite-sample correction
      p = (1 + #{|null| >= |observed|}) / (1 + n_iterations),
  giving a non-zero floor of 1 / (n_iterations + 1).
"""

from __future__ import annotations

import numpy as np


def matched_pair_permutation_test(
    contradiction: np.ndarray,
    coherent_edit: np.ndarray,
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict:
    """Two-sided within-pair sign-flip permutation test on mean contrast.

    Args:
        contradiction: 1D array of M_k scalars, one per pair (contradiction
            member).
        coherent_edit: 1D array of M_k scalars, pair-aligned (coherent-edit
            control member).
        n_iterations: number of sign-flip permutations. Default 10,000.
        seed: RNG seed for reproducibility.

    Returns:
        dict with:
          - 'observed': float, mean of Delta = contradiction - coherent_edit
          - 'null_distribution': ndarray of n_iterations permuted means
          - 'p_value': float, two-sided with (1+count)/(1+n) correction
          - 'n_pairs': int

    Raises:
        ValueError: if inputs misalign, are empty, or n_iterations <= 0.
    """
    c = np.asarray(contradiction, dtype=np.float64).ravel()
    e = np.asarray(coherent_edit, dtype=np.float64).ravel()
    if c.shape != e.shape:
        raise ValueError(
            f"contradiction and coherent_edit must align; got {c.shape} and {e.shape}"
        )
    if c.size == 0:
        raise ValueError("inputs must be non-empty")
    if n_iterations <= 0:
        raise ValueError(f"n_iterations must be > 0, got {n_iterations}")

    delta = c - e
    n_pairs = delta.size
    observed = float(np.mean(delta))

    rng = np.random.default_rng(seed)
    # signs: shape (n_iterations, n_pairs), each entry +1 or -1.
    signs = rng.integers(0, 2, size=(n_iterations, n_pairs)) * 2 - 1
    null_means = (signs * delta).mean(axis=1)

    abs_obs = abs(observed)
    count_extreme = int(np.sum(np.abs(null_means) >= abs_obs))
    p_value = (1 + count_extreme) / (1 + n_iterations)

    return {
        "observed": observed,
        "null_distribution": null_means,
        "p_value": float(p_value),
        "n_pairs": int(n_pairs),
    }


# ----- Sanity check (run: python -m src.analysis.permutation) -----


def _sanity_check() -> None:
    import sys

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        marker = "PASS" if cond else "FAIL"
        tail = f": {detail}" if (detail and not cond) else ""
        print(f"  [{marker}] {name}{tail}")
        if not cond:
            failures.append(name)

    rng = np.random.default_rng(0)
    n = 15

    # Case 1: identical members -> observed 0, p = 1.0 (no false positive).
    contra = 1.0 + rng.normal(0, 0.05, n)
    res_null = matched_pair_permutation_test(contra, contra.copy(), n_iterations=500, seed=1)
    check(
        "identical members: observed mean is 0",
        abs(res_null["observed"]) < 1e-12,
        detail=f"got {res_null['observed']}",
    )
    check(
        "identical members: p-value not significant",
        res_null["p_value"] > 0.05,
        detail=f"got p={res_null['p_value']:.4f}",
    )

    # Case 2: strongly separated, all deltas large positive -> tiny p.
    contra2 = 5.0 + rng.normal(0, 0.05, n)
    edit2 = 1.0 + rng.normal(0, 0.05, n)
    res_sep = matched_pair_permutation_test(contra2, edit2, n_iterations=500, seed=1)
    check(
        "separated: observed mean ~ 4",
        abs(res_sep["observed"] - 4.0) < 0.3,
        detail=f"got {res_sep['observed']:.3f}",
    )
    check(
        "separated: p-value < 0.05",
        res_sep["p_value"] < 0.05,
        detail=f"got p={res_sep['p_value']:.4f}",
    )

    # Case 3: null distribution is symmetric about 0 (mean ~ 0).
    check(
        "null distribution centered near 0",
        abs(float(np.mean(res_sep["null_distribution"]))) < 0.5,
        detail=f"got {float(np.mean(res_sep['null_distribution'])):.3f}",
    )

    # Case 4: null distribution length.
    check("null_distribution has correct length", len(res_sep["null_distribution"]) == 500)

    # Case 5: p-value bounds and floor.
    check("p-value in (0, 1]", 0 < res_sep["p_value"] <= 1.0, detail=f"got {res_sep['p_value']}")
    check(
        "p-value floor = 1/(n+1)",
        res_sep["p_value"] >= 1.0 / 501,
        detail=f"got {res_sep['p_value']}, floor {1/501:.6f}",
    )

    # Case 6: sign of observed does not change the p-value (two-sided).
    res_flip = matched_pair_permutation_test(edit2, contra2, n_iterations=500, seed=1)
    check(
        "two-sided: flipping members negates observed",
        abs(res_flip["observed"] + res_sep["observed"]) < 1e-9,
        detail=f"got {res_flip['observed']:.3f} vs {res_sep['observed']:.3f}",
    )
    check(
        "two-sided: p-value unchanged by member order",
        res_flip["p_value"] == res_sep["p_value"],
        detail=f"got {res_flip['p_value']} vs {res_sep['p_value']}",
    )

    # Case 7: reproducibility under fixed seed.
    ra = matched_pair_permutation_test(contra2, edit2, n_iterations=200, seed=42)
    rb = matched_pair_permutation_test(contra2, edit2, n_iterations=200, seed=42)
    check("same seed gives same p-value", ra["p_value"] == rb["p_value"])
    check(
        "same seed gives same null distribution",
        np.array_equal(ra["null_distribution"], rb["null_distribution"]),
    )

    # Case 8: different seed gives different null distribution.
    rc = matched_pair_permutation_test(contra2, edit2, n_iterations=200, seed=99)
    check(
        "different seed gives different null distribution",
        not np.array_equal(ra["null_distribution"], rc["null_distribution"]),
    )

    # Case 9: input validation.
    try:
        matched_pair_permutation_test(np.array([]), np.array([]), n_iterations=10)
        check("empty inputs rejected", False)
    except ValueError:
        check("empty inputs rejected with ValueError", True)
    try:
        matched_pair_permutation_test(np.array([1.0, 2.0]), np.array([1.0]), n_iterations=10)
        check("misaligned inputs rejected", False)
    except ValueError:
        check("misaligned inputs rejected with ValueError", True)
    try:
        matched_pair_permutation_test(contra2, edit2, n_iterations=0)
        check("n_iterations=0 rejected", False)
    except ValueError:
        check("n_iterations=0 rejected with ValueError", True)

    # Case 10: result structure.
    check(
        "result has expected keys",
        set(res_sep.keys()) == {"observed", "null_distribution", "p_value", "n_pairs"},
    )
    check("n_pairs reported correctly", res_sep["n_pairs"] == n)

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {failures}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    print("permutation sanity check\n")
    _sanity_check()
