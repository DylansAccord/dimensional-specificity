"""
Position-only baselines and the necessity gate (analysis_protocol Section 14).

These baselines read POSITION (where the token embeddings sit), not SHAPE (the
topology of the cloud). Per the standing principle in Section 0, a topological
result counts only if it beats these on the identical pairs: if a linear probe on
the pooled embeddings separates contradiction from the coherent-edit control as
well as the H_k void does, the topology is decoration, not evidence.

Every function operates on arrays so the logic is testable without file I/O. A
driver supplies, for the identical items the topology uses:
  - X: pooled per-item embeddings, shape (n_items, hidden_size)
  - y: 1 for contradiction items, 0 for coherent-edit control items
  - groups: source-family (or pair) id per item, so cross-validation does not
    leak a source across folds
  - mk_scores: the per-item topological scalar M_k* (for the incremental test)
  - topo_auc: the topology's own discrimination AUC on these items
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NECESSITY_MARGIN = 0.05  # AUC; topology must beat the best position baseline by this


def auc_from_scores(scores, y) -> float:
    """AUC of a per-item scalar discriminating y (1=contradiction, 0=control)."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, scores))


def _grouped_cv_auc(estimator, X, y, groups, n_splits: int = 5) -> float:
    """Cross-validated AUC with grouped folds, so no source leaks across folds."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2 or len(np.unique(y)) < 2:
        return 0.5
    cv = GroupKFold(n_splits=n_splits)
    proba = cross_val_predict(
        estimator, X, y, cv=cv, groups=groups, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(y, proba))


def probe_auc(X, y, groups, n_splits: int = 5, C: float = 1.0) -> float:
    """L2 logistic regression on the pooled embeddings; grouped-CV AUC.

    This is the rigorous position baseline: the held-out ability of a linear
    classifier to separate the classes from position alone.
    """
    est = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return _grouped_cv_auc(est, X, y, groups, n_splits)


def pca_probe_auc(X, y, groups, n_components: int = 20, n_splits: int = 5) -> float:
    """Linear-subspace baseline: top principal components, then logistic regression."""
    X = np.asarray(X, dtype=float)
    nc = max(1, min(n_components, X.shape[1], X.shape[0] - 1))
    est = make_pipeline(
        StandardScaler(), PCA(n_components=nc), LogisticRegression(max_iter=2000)
    )
    return _grouped_cv_auc(est, X, y, groups, n_splits)


def centroid_auc(X, y) -> float:
    """Cheap position baseline: per-item score = distance to control centroid minus
    distance to contradiction centroid. In-sample (centroids from all items), so it
    is an optimistic floor reported for context, not a held-out estimate. The probe
    is the rigorous baseline."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return 0.5
    c1 = X[y == 1].mean(axis=0)
    c0 = X[y == 0].mean(axis=0)
    d1 = np.linalg.norm(X - c1, axis=1)
    d0 = np.linalg.norm(X - c0, axis=1)
    return auc_from_scores(d0 - d1, y)  # higher => nearer the contradiction centroid


def position_baselines(X, y, groups, pca_components: int = 20) -> dict:
    """All position-only baseline AUCs on the identical items."""
    return {
        "linear_probe": probe_auc(X, y, groups),
        "pca_probe": pca_probe_auc(X, y, groups, n_components=pca_components),
        "centroid": centroid_auc(X, y),
    }


def incremental_auc(X, mk_scores, y, groups, n_splits: int = 5) -> dict:
    """Nested comparison: AUC(position) vs AUC(position + M_k*). The topological
    feature is necessary only if it lifts discrimination beyond position alone."""
    X = np.asarray(X, dtype=float)
    mk = np.asarray(mk_scores, dtype=float).reshape(-1, 1)
    base = probe_auc(X, y, groups, n_splits)
    full = probe_auc(np.hstack([X, mk]), y, groups, n_splits)
    return {"baseline_auc": base, "with_topology_auc": full, "lift": full - base}


def necessity_verdict(topo_auc, baseline_aucs: dict, margin: float = NECESSITY_MARGIN) -> dict:
    """Section 14 gate: the topology must beat the best position baseline by >= margin."""
    best_name = max(baseline_aucs, key=baseline_aucs.get)
    best = baseline_aucs[best_name]
    return {
        "topology_auc": float(topo_auc),
        "best_baseline": best_name,
        "best_baseline_auc": float(best),
        "margin": margin,
        "topology_exceeds_baselines": bool((float(topo_auc) - float(best)) >= margin),
        "all_baselines": {k: float(v) for k, v in baseline_aucs.items()},
    }


# ----- Sanity check (run: python -m src.analysis.baselines) -----


def _sanity_check() -> None:
    import sys

    failures: list[str] = []

    def check(name, cond, detail=""):
        marker = "PASS" if cond else "FAIL"
        tail = f": {detail}" if (detail and not cond) else ""
        print(f"  [{marker}] {name}{tail}")
        if not cond:
            failures.append(name)

    rng = np.random.default_rng(0)
    n_pairs = 60
    dim = 32
    # Paired layout: n_pairs families, each a (contradiction, control) pair. y
    # alternates 1, 0; groups give each pair its own family id so grouped CV never
    # trains and tests on the same family.
    y = np.array([1, 0] * n_pairs)
    groups = np.repeat(np.arange(n_pairs), 2)

    # auc_from_scores correctness.
    yy = np.array([1, 1, 0, 0])
    check("perfectly separating scores -> AUC 1.0", auc_from_scores([3, 2, 1, 0], yy) == 1.0)
    check("reversed scores -> AUC 0.0", auc_from_scores([0, 1, 2, 3], yy) == 0.0)
    check("single class -> AUC 0.5", auc_from_scores([1, 2, 3], [1, 1, 1]) == 0.5)

    # Scenario A: POSITION-separable. A linear direction separates the classes,
    # M_k is pure noise. The probe should score high and topology should NOT beat it.
    base = rng.normal(size=(2 * n_pairs, dim))
    direction = np.zeros(dim); direction[0] = 1.0
    X_pos = base + np.outer((y == 1).astype(float) * 3.0, direction)   # class-1 shifted
    mk_noise = rng.normal(size=2 * n_pairs)
    bl_A = position_baselines(X_pos, y, groups)
    topo_auc_A = auc_from_scores(mk_noise, y)            # ~0.5
    verdict_A = necessity_verdict(topo_auc_A, bl_A)
    check("position-separable: linear probe AUC high (>0.9)",
          bl_A["linear_probe"] > 0.9, detail=f"{bl_A['linear_probe']:.3f}")
    check("position-separable: topology does NOT pass necessity gate",
          verdict_A["topology_exceeds_baselines"] is False,
          detail=f"topo {topo_auc_A:.3f} vs best {verdict_A['best_baseline_auc']:.3f}")

    # Scenario B: POSITION-blind but TOPOLOGY-separable. Embeddings carry no linear
    # class signal (probe ~0.5), but M_k cleanly separates. Topology should pass.
    X_blind = rng.normal(size=(2 * n_pairs, dim))          # no class signal in position
    mk_signal = (y == 1).astype(float) * 2.0 + rng.normal(0, 0.3, size=2 * n_pairs)
    bl_B = position_baselines(X_blind, y, groups)
    topo_auc_B = auc_from_scores(mk_signal, y)           # ~high
    verdict_B = necessity_verdict(topo_auc_B, bl_B)
    check("position-blind: linear probe near chance (<0.65)",
          bl_B["linear_probe"] < 0.65, detail=f"{bl_B['linear_probe']:.3f}")
    check("position-blind: topology PASSES necessity gate",
          verdict_B["topology_exceeds_baselines"] is True,
          detail=f"topo {topo_auc_B:.3f} vs best {verdict_B['best_baseline_auc']:.3f}")

    # Incremental validity: in Scenario B, M_k* should lift the probe meaningfully.
    inc_B = incremental_auc(X_blind, mk_signal, y, groups)
    check("incremental: M_k lifts AUC when it carries signal (lift > 0.1)",
          inc_B["lift"] > 0.1, detail=f"lift {inc_B['lift']:.3f}")
    # In Scenario A, M_k noise should not lift much.
    inc_A = incremental_auc(X_pos, mk_noise, y, groups)
    check("incremental: M_k noise gives ~no lift (lift < 0.1)",
          inc_A["lift"] < 0.1, detail=f"lift {inc_A['lift']:.3f}")

    # necessity_verdict mechanics.
    v = necessity_verdict(0.80, {"linear_probe": 0.74, "pca_probe": 0.70, "centroid": 0.60})
    check("verdict picks best baseline", v["best_baseline"] == "linear_probe")
    check("verdict margin met (0.80 - 0.74 >= 0.05)", v["topology_exceeds_baselines"] is True)
    v2 = necessity_verdict(0.78, {"linear_probe": 0.74})
    check("verdict margin NOT met (0.78 - 0.74 < 0.05)", v2["topology_exceeds_baselines"] is False)

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {failures}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    print("baselines sanity check\n")
    _sanity_check()
