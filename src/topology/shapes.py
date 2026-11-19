"""
Canonical-shape generators for sanity-check validation.

Three shapes with known topology, used to verify the persistent homology
pipeline correctly distinguishes topology types via bottleneck distance:

  Circle (S^1):  Betti-0 = 1, Betti-1 = 1, Betti-2 = 0
  Sphere (S^2):  Betti-0 = 1, Betti-1 = 0, Betti-2 = 1
  Torus  (T^2):  Betti-0 = 1, Betti-1 = 2, Betti-2 = 1

Each is sampled in its natural ambient dimension (2D or 3D) and then embedded
via random orthonormal projection into the LLM hidden dimension (4096 for
Llama-3.1-8B) to confirm the pipeline operates correctly at production dim.
"""

from __future__ import annotations

import numpy as np


def sample_circle(n: int = 200, radius: float = 1.0, noise: float = 0.02,
                  seed: int = 0) -> np.ndarray:
    """N points on a unit circle in R^2 with small gaussian noise."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    x = radius * np.cos(theta) + rng.normal(0, noise, n)
    y = radius * np.sin(theta) + rng.normal(0, noise, n)
    return np.column_stack([x, y])


def sample_sphere(n: int = 400, radius: float = 1.0, noise: float = 0.02,
                  seed: int = 0) -> np.ndarray:
    """N points uniformly on a unit 2-sphere in R^3 via gaussian normalization."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * radius
    pts += rng.normal(0, noise, pts.shape)
    return pts


def sample_torus(n: int = 500, R: float = 2.0, r: float = 0.7,
                 noise: float = 0.02, seed: int = 0) -> np.ndarray:
    """N points on a torus in R^3 with major radius R and minor radius r."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    pts = np.column_stack([x, y, z])
    pts += rng.normal(0, noise, pts.shape)
    return pts


def embed_in_high_dim(pts: np.ndarray, target_dim: int = 4096,
                     seed: int = 0) -> np.ndarray:
    """
    Random orthonormal embedding into target_dim.

    Orthonormal rotation is an isometry, so distances (and therefore the
    persistence diagram) are preserved exactly. Used to verify the pipeline
    operates correctly at the LLM hidden dimension (Llama-3.1-8B: 4096).
    """
    rng = np.random.default_rng(seed)
    n, d = pts.shape
    if target_dim <= d:
        return pts
    padded = np.zeros((n, target_dim))
    padded[:, :d] = pts
    A = rng.normal(0, 1, (target_dim, target_dim))
    Q, _ = np.linalg.qr(A)
    return padded @ Q
