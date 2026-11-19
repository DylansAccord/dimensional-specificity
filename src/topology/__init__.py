"""Persistent homology and zigzag persistence pipeline.

Two arms:
  - Vietoris-Rips persistence on point clouds (ripser) — sentence-embedding arm
  - Zigzag persistence on graph sequences (dionysus) — attention-graph arm

Bottleneck distance is the primary statistical comparison method, following
the validated approach from the Saturday probe: exact Betti number recovery
from noisy point clouds is brittle, but bottleneck distance between
persistence diagrams reliably discriminates topology.
"""
