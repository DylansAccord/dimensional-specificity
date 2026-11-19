"""
Sanity check: validate that the persistent homology pipeline can DISTINGUISH
shapes with different topology, as measured by bottleneck distance between
persistence diagrams.

This is the critical gate that must pass before any language-model experiment
runs. The check ports the validated approach from the Saturday probe
(murmur-probe/src/sanity_check.py) and extends it to the LLM hidden dimension.

Why bottleneck distance and not exact Betti number recovery:

Exact Betti number recovery from noisy point clouds is brittle. The Saturday
probe established this empirically: even on a canonical torus (B1=2, B2=1),
the two H1 features have very different persistences (1.895 and 0.789, a
2.4x ratio), so no single threshold reliably catches both as signal while
rejecting noise. Gap-based detection, persistence-multiplier thresholds, and
the various standard heuristics each fail on at least one canonical shape.

Bottleneck distance does not require choosing a threshold. It is stable
under small perturbations of the data and is the standard TDA metric for
quantifying how topologically different two point clouds are. The Saturday
probe demonstrated 11-20x discrimination ratios between different-topology
and same-topology comparisons in R^768; this module verifies the same in
R^4096 (Llama-3.1-8B hidden dim).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ripser import ripser

from .distances import bottleneck_distance
from .shapes import (
    embed_in_high_dim,
    sample_circle,
    sample_sphere,
    sample_torus,
)


@dataclass
class SanityCheckResult:
    """Outcome of the bottleneck-discrimination sanity check."""
    comparisons: list[dict]              # individual pairwise comparisons
    passed_by_dim: dict[int, dict]       # per-dimension pass/fail summary
    passed: bool                          # overall: all dims must pass
    target_dim: int                       # ambient embedding dimension


def run_sanity_check(target_dim: int = 4096,
                     ratio_threshold: float = 5.0,
                     verbose: bool = True) -> SanityCheckResult:
    """
    Compute bottleneck distances between persistence diagrams of canonical shapes.

    Pass criterion: at each dim in {1, 2}, all different-topology distances
    must be at least `ratio_threshold` times larger than the same-topology
    distance. Default ratio_threshold of 5.0 is conservative; Saturday probe
    achieved 11-20x in R^768.
    """
    shapes = {
        "circle":   sample_circle(n=200, seed=0),
        "circle_b": sample_circle(n=200, seed=1),  # different sample, same topology
        "sphere":   sample_sphere(n=400, seed=0),
        "torus":    sample_torus(n=500, seed=0),
    }
    # Embed all into high dim. Distances preserved exactly (rotation is isometry).
    shapes = {name: embed_in_high_dim(pts, target_dim=target_dim, seed=42)
              for name, pts in shapes.items()}

    diagrams = {}
    for name, pts in shapes.items():
        if verbose:
            print(f"  computing diagrams for {name} ({pts.shape[0]} pts in R^{target_dim})...")
        diagrams[name] = ripser(pts, maxdim=2)["dgms"]

    comparisons_spec = [
        ("circle", "circle_b", "same",       1),
        ("circle", "circle_b", "same",       2),
        ("circle", "sphere",   "different",  1),
        ("circle", "sphere",   "different",  2),
        ("circle", "torus",    "different",  1),
        ("circle", "torus",    "different",  2),
        ("sphere", "torus",    "different",  1),
        ("sphere", "torus",    "different",  2),
    ]

    comparisons = []
    for a, b, kind, dim in comparisons_spec:
        d = bottleneck_distance(diagrams[a][dim], diagrams[b][dim])
        comparisons.append({"a": a, "b": b, "kind": kind, "dim": dim, "distance": d})
        if verbose:
            print(f"    H{dim}  {kind:9s}  {a:9s} vs {b:9s}  d = {d:.4f}")

    passed_by_dim = {}
    overall_passed = True
    for dim in (1, 2):
        same = [r["distance"] for r in comparisons
                if r["kind"] == "same" and r["dim"] == dim]
        diff = [r["distance"] for r in comparisons
                if r["kind"] == "different" and r["dim"] == dim]
        max_same = max(same) if same else 0.0
        min_diff = min(diff) if diff else float("inf")
        ratio = min_diff / max(max_same, 1e-6)
        passed = ratio >= ratio_threshold
        passed_by_dim[dim] = {
            "max_same": max_same,
            "min_diff": min_diff,
            "ratio": ratio,
            "passed": passed,
        }
        if not passed:
            overall_passed = False
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  H{dim}: max(same)={max_same:.4f}  min(diff)={min_diff:.4f}  "
                  f"ratio={ratio:.1f}x  {status}")

    return SanityCheckResult(
        comparisons=comparisons,
        passed_by_dim=passed_by_dim,
        passed=overall_passed,
        target_dim=target_dim,
    )
