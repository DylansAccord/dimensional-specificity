"""
Dimensional dominance test and Holm-Bonferroni multiple-comparison correction.

This is where the Murmur hypothesis gets tested. Given Cohen's d_z effect
sizes and permutation p-values per dimension for a fixed (arm, model) cell,
dimensional_dominance_test asks whether a single dimension k* in the search
window {2, 3, 4} dominates the rest of the window.

Per docs/analysis_protocol.md Section 5, dominance by effect size requires
BOTH:
  - margin: d_z(k*) exceeds the second-largest d_z WITHIN the window by at
    least `margin` (default 0.2);
  - floor: d_z(k*) is at least `floor` (default 0.5, a medium effect).

The third dominance condition in the protocol, Holm-Bonferroni significance
of the permutation p-value for k*, is applied at the scripts/04 level, where
the full family of p-values is available. This module reports the effect-size
dominance (is_dominant) and echoes the p-values so 04 can AND in the
corrected significance.

Adjacent dimensions are reported separately, not folded into the margin:
  - h1_check: H1 effect and p-value, plus a flag for whether H1 exceeds the
    in-window best. H1 exceedance is the Section 6.1 secondary finding that
    falsifies dimensional-specificity as framed without falsifying the
    architecture. It is deliberately NOT part of the in-window margin.
  - h0_check: H0 effect and p-value, the Jiang-aligned baseline (Section 6.2).

holm_bonferroni_correction is family-agnostic: hand it any flat list or dict
of p-values, get back adjusted values and rejection decisions.
"""

from __future__ import annotations

from typing import Hashable

import numpy as np


def holm_bonferroni_correction(
    p_values: list[float] | dict[Hashable, float],
    alpha: float = 0.05,
) -> dict:
    """Holm-Bonferroni step-down correction for FWER control.

    Args:
        p_values: list or dict of p-values.
        alpha: target family-wise error rate.

    Returns:
        dict with:
          - 'adjusted_p_values': same shape as input
          - 'rejected': same shape, booleans (True = reject H_i)
          - 'n_rejected': int

    Algorithm:
        1. Sort p-values ascending.
        2. For sorted index i (1-indexed): unbounded_(i) = (m - i + 1) * p_(i).
        3. adjusted_(i) = min(1, max over j <= i of unbounded_(j)).
        4. Reject H_(i) iff adjusted_(i) <= alpha.
        5. Map results back to the original input order.
    """
    if isinstance(p_values, dict):
        keys = list(p_values.keys())
        pvals = np.array([p_values[k] for k in keys], dtype=np.float64)
        return_dict = True
    else:
        keys = None
        pvals = np.asarray(p_values, dtype=np.float64)
        return_dict = False

    m = len(pvals)
    if m == 0:
        if return_dict:
            return {"adjusted_p_values": {}, "rejected": {}, "n_rejected": 0}
        return {"adjusted_p_values": [], "rejected": [], "n_rejected": 0}

    sort_order = np.argsort(pvals, kind="stable")
    sorted_p = pvals[sort_order]
    multipliers = np.arange(m, 0, -1, dtype=np.float64)  # m, m-1, ..., 1

    unbounded = sorted_p * multipliers
    adjusted_sorted = np.minimum(np.maximum.accumulate(unbounded), 1.0)
    rejected_sorted = adjusted_sorted <= alpha

    adjusted = np.empty(m, dtype=np.float64)
    rejected = np.empty(m, dtype=bool)
    adjusted[sort_order] = adjusted_sorted
    rejected[sort_order] = rejected_sorted
    n_rejected = int(rejected.sum())

    if return_dict:
        return {
            "adjusted_p_values": {keys[i]: float(adjusted[i]) for i in range(m)},
            "rejected": {keys[i]: bool(rejected[i]) for i in range(m)},
            "n_rejected": n_rejected,
        }
    return {
        "adjusted_p_values": adjusted.tolist(),
        "rejected": rejected.tolist(),
        "n_rejected": n_rejected,
    }


def dimensional_dominance_test(
    effect_sizes: dict[int, float],
    p_values: dict[int, float],
    margin: float = 0.2,
    floor: float = 0.5,
    search_window: tuple[int, ...] = (2, 3, 4),
) -> dict:
    """Test whether a single in-window dimension dominates by effect size.

    Args:
        effect_sizes: dict mapping dimension k to Cohen's d_z.
        p_values: dict mapping dimension k to permutation p-value. Keys must
            match effect_sizes.
        margin: minimum d_z gap over the second-best dim IN the window.
            Default 0.2 per the protocol.
        floor: minimum d_z for the winning dim. Default 0.5 (medium effect).
        search_window: dimensions where the Murmur prediction allows the
            dominant dim to lie. Default (2, 3, 4).

    Returns:
        dict containing:
          - 'effect_sizes', 'p_values', 'search_window' echoed
          - 'margin_threshold', 'floor_threshold' echoed
          - 'best_in_search_window':
              'dim', 'effect_size', 'p_value',
              'second_best_in_window', 'margin_over_window',
              'meets_margin', 'meets_floor', 'is_dominant'
          - 'h1_check': 'effect_size', 'p_value', 'exceeds_search_window_best'
          - 'h0_check': 'effect_size', 'p_value'

    Raises:
        ValueError: if effect_sizes and p_values have mismatched keys, or if
            any search_window dim is missing from effect_sizes.
    """
    if set(effect_sizes.keys()) != set(p_values.keys()):
        raise ValueError(
            "effect_sizes and p_values must have the same keys; got "
            f"effect_sizes={sorted(effect_sizes)}, p_values={sorted(p_values)}"
        )
    missing = [k for k in search_window if k not in effect_sizes]
    if missing:
        raise ValueError(f"search_window dims missing from effect_sizes: {missing}")

    # Best dim within the window.
    best_dim = max(search_window, key=lambda k: effect_sizes[k])
    best_effect = effect_sizes[best_dim]

    # Margin is measured against the runner-up WITHIN the window only.
    in_window_others = [effect_sizes[k] for k in search_window if k != best_dim]
    if in_window_others:
        second_best = max(in_window_others)
        margin_over_window = best_effect - second_best
    else:
        second_best = float("nan")
        margin_over_window = best_effect  # degenerate single-dim window

    meets_margin = margin_over_window >= margin
    meets_floor = best_effect >= floor
    is_dominant = bool(meets_margin and meets_floor)

    best_in_search_window = {
        "dim": int(best_dim),
        "effect_size": float(best_effect),
        "p_value": float(p_values[best_dim]),
        "second_best_in_window": float(second_best),
        "margin_over_window": float(margin_over_window),
        "meets_margin": bool(meets_margin),
        "meets_floor": bool(meets_floor),
        "is_dominant": is_dominant,
    }

    h1_check = {
        "effect_size": float(effect_sizes.get(1, float("nan"))),
        "p_value": float(p_values.get(1, float("nan"))),
        "exceeds_search_window_best": bool(
            1 in effect_sizes and effect_sizes[1] > best_effect
        ),
    }

    h0_check = {
        "effect_size": float(effect_sizes.get(0, float("nan"))),
        "p_value": float(p_values.get(0, float("nan"))),
    }

    return {
        "effect_sizes": {int(k): float(v) for k, v in effect_sizes.items()},
        "p_values": {int(k): float(v) for k, v in p_values.items()},
        "search_window": list(search_window),
        "margin_threshold": float(margin),
        "floor_threshold": float(floor),
        "best_in_search_window": best_in_search_window,
        "h1_check": h1_check,
        "h0_check": h0_check,
    }


# ----- Sanity check (run: python -m src.analysis.dominance) -----


def _sanity_check() -> None:
    import sys

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        marker = "PASS" if cond else "FAIL"
        tail = f": {detail}" if (detail and not cond) else ""
        print(f"  [{marker}] {name}{tail}")
        if not cond:
            failures.append(name)

    # --- Holm-Bonferroni (unchanged, revalidated) ---

    # Case 1: textbook example. p = [0.01, 0.03, 0.04], alpha = 0.05.
    # unbounded sorted: [3*0.01, 2*0.03, 1*0.04] = [0.03, 0.06, 0.04]
    # cummax: [0.03, 0.06, 0.06]; reject: [True, False, False].
    res = holm_bonferroni_correction([0.01, 0.03, 0.04], alpha=0.05)
    check(
        "Holm: [0.01,0.03,0.04] adjusted = [0.03,0.06,0.06]",
        all(abs(a - b) < 1e-12 for a, b in zip(res["adjusted_p_values"], [0.03, 0.06, 0.06])),
        detail=f"got {res['adjusted_p_values']}",
    )
    check("Holm: rejects only first", res["rejected"] == [True, False, False])

    # Case 2: all small.
    res = holm_bonferroni_correction([0.001, 0.002, 0.003], alpha=0.05)
    check("Holm: all-small rejects all three", res["rejected"] == [True, True, True])

    # Case 3: dict input returns dict and preserves keys.
    res = holm_bonferroni_correction({"a": 0.001, "b": 0.5, "c": 0.6}, alpha=0.05)
    check(
        "Holm: dict input rejects only 'a'",
        res["rejected"] == {"a": True, "b": False, "c": False},
    )

    # Case 4: order preserved under unsorted input.
    res = holm_bonferroni_correction([0.5, 0.001, 0.6], alpha=0.05)
    check("Holm: rejection at original position of smallest", res["rejected"] == [False, True, False])

    # Case 5: empty input.
    res = holm_bonferroni_correction([], alpha=0.05)
    check(
        "Holm: empty input returns empty results",
        res["adjusted_p_values"] == [] and res["rejected"] == [] and res["n_rejected"] == 0,
    )

    # --- Dominance ---

    # Case 6: clear winner, above floor, margin within window.
    # best = dim2 (1.5); in-window runner-up = max(0.4,0.3) = 0.4;
    # margin = 1.1 >= 0.2; floor 1.5 >= 0.5 -> is_dominant True.
    es = {0: 0.1, 1: 0.2, 2: 1.5, 3: 0.4, 4: 0.3}
    pv = {0: 0.5, 1: 0.4, 2: 0.001, 3: 0.3, 4: 0.2}
    res = dimensional_dominance_test(es, pv)
    b = res["best_in_search_window"]
    check("Dominance: winner is dim 2", b["dim"] == 2, detail=f"got {b['dim']}")
    check(
        "Dominance: in-window margin = 1.5 - 0.4 = 1.1",
        abs(b["margin_over_window"] - 1.1) < 1e-9,
        detail=f"got {b['margin_over_window']}",
    )
    check("Dominance: is_dominant True", b["is_dominant"])

    # Case 7: winner but BELOW floor. best = dim2 (0.45); margin 0.35 ok but
    # floor 0.45 < 0.5 -> meets_margin True, meets_floor False, not dominant.
    es = {0: 0.05, 1: 0.05, 2: 0.45, 3: 0.1, 4: 0.1}
    pv = {0: 0.5, 1: 0.5, 2: 0.02, 3: 0.3, 4: 0.3}
    res = dimensional_dominance_test(es, pv)
    b = res["best_in_search_window"]
    check("Dominance: below-floor meets_margin True", b["meets_margin"])
    check("Dominance: below-floor meets_floor False", not b["meets_floor"])
    check("Dominance: below-floor is_dominant False", not b["is_dominant"])

    # Case 8: smeared. best = dim2 (0.7), runner-up 0.6 -> margin 0.1 < 0.2.
    # floor ok but margin fails -> not dominant.
    es = {0: 0.1, 1: 0.1, 2: 0.7, 3: 0.6, 4: 0.55}
    pv = {0: 0.5, 1: 0.5, 2: 0.05, 3: 0.05, 4: 0.05}
    res = dimensional_dominance_test(es, pv)
    b = res["best_in_search_window"]
    check(
        "Dominance: smeared margin = 0.7 - 0.6 = 0.1",
        abs(b["margin_over_window"] - 0.1) < 1e-9,
        detail=f"got {b['margin_over_window']}",
    )
    check("Dominance: smeared meets_margin False", not b["meets_margin"])
    check("Dominance: smeared is_dominant False", not b["is_dominant"])

    # Case 9: H1 exceeds the window. In-window dominance still holds (dim2
    # over dim3 by 0.3, above floor), but h1_check flags the exceedance.
    # These are separate concerns by design.
    es = {0: 0.1, 1: 2.0, 2: 0.7, 3: 0.4, 4: 0.3}
    pv = {0: 0.5, 1: 0.001, 2: 0.02, 3: 0.3, 4: 0.3}
    res = dimensional_dominance_test(es, pv)
    b = res["best_in_search_window"]
    check(
        "Dominance: in-window margin ignores H1 (0.7 - 0.4 = 0.3)",
        abs(b["margin_over_window"] - 0.3) < 1e-9,
        detail=f"got {b['margin_over_window']}",
    )
    check("Dominance: in-window is_dominant True", b["is_dominant"])
    check(
        "Dominance: h1 exceedance flagged",
        res["h1_check"]["exceeds_search_window_best"],
    )

    # Case 10: H0 (Jiang-aligned) effect reported.
    check("Dominance: H0 effect reported = 0.1", abs(res["h0_check"]["effect_size"] - 0.1) < 1e-9)

    # Case 11: thresholds echoed.
    check(
        "Dominance: default thresholds margin 0.2 floor 0.5",
        abs(res["margin_threshold"] - 0.2) < 1e-12 and abs(res["floor_threshold"] - 0.5) < 1e-12,
    )

    # Case 12: result structure.
    expected_top = {
        "effect_sizes", "p_values", "search_window", "margin_threshold",
        "floor_threshold", "best_in_search_window", "h1_check", "h0_check",
    }
    check("Dominance: top-level keys present", set(res.keys()) == expected_top, detail=f"got {sorted(res.keys())}")
    expected_best = {
        "dim", "effect_size", "p_value", "second_best_in_window",
        "margin_over_window", "meets_margin", "meets_floor", "is_dominant",
    }
    check(
        "Dominance: best_in_search_window keys present",
        set(res["best_in_search_window"].keys()) == expected_best,
        detail=f"got {sorted(res['best_in_search_window'].keys())}",
    )

    # Case 13: input validation.
    try:
        dimensional_dominance_test({0: 0.1, 1: 0.2}, {0: 0.5})
        check("Dominance: mismatched keys rejected", False)
    except ValueError:
        check("Dominance: mismatched keys rejected with ValueError", True)
    try:
        dimensional_dominance_test({0: 0.1}, {0: 0.5}, search_window=(2, 3, 4))
        check("Dominance: missing window dim rejected", False)
    except ValueError:
        check("Dominance: missing window dim rejected with ValueError", True)

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {failures}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    print("dominance sanity check\n")
    _sanity_check()
