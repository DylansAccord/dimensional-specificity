#!/usr/bin/env python
"""
Phase 0: Topology pipeline sanity check.

Validates that ripser + persim can distinguish shapes with different topology
in the Llama-3-8B hidden dimension (R^4096). This MUST pass before proceeding
to the language-model experiment.

Pass criterion: bottleneck distance between different-topology shapes is at
least 5x larger than between different samples of the same shape, at both
H1 and H2.

Cost: $0, runs locally in 2-5 minutes.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.topology.sanity_check import run_sanity_check


def main():
    print("=" * 70)
    print("Murmur Dimensional-Specificity Experiment — Phase 0")
    print("Topology Pipeline Sanity Check")
    print("=" * 70)
    print()
    print("Validating ripser + persim can distinguish shapes with different")
    print("topology, as measured by bottleneck distance between persistence")
    print("diagrams.")
    print()
    print("Each canonical shape is embedded in R^4096 to match the Llama-3-8B")
    print("hidden dimension where actual topology computation will occur.")
    print()

    result = run_sanity_check(target_dim=4096, ratio_threshold=5.0, verbose=True)

    print()
    if result.passed:
        print("PASS: pipeline correctly distinguishes topology types in R^4096.")
        print(f"      H1 discrimination ratio: {result.passed_by_dim[1]['ratio']:.1f}x")
        print(f"      H2 discrimination ratio: {result.passed_by_dim[2]['ratio']:.1f}x")
        print("      You may proceed to scripts/01_build_corpus.py")
        sys.exit(0)
    else:
        print("FAIL: pipeline cannot reliably distinguish topology.")
        print("      Do NOT proceed to the full experiment.")
        print("      Check ripser and persim installations and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
