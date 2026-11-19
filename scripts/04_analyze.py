#!/usr/bin/env python
"""
Phase 4: Statistical analysis of the dimensional-specificity experiment.

Consumes the topology stage outputs (topology_manifest.json plus the per-item
.npz diagrams) and the corpus pairing (pairs.jsonl from scripts/01), forms
matched pairs, and runs the preregistered analysis in docs/analysis_protocol.md.

Per pair p and dimension k the signed contrast is
    Delta_k(p) = M_k(contradiction_p) - M_k(control_p),
where M_k is max persistence (src.analysis.bootstrap.max_persistence). The
per-dimension effect size is the paired Cohen's d_z, with a percentile
bootstrap CI over pairs and a within-pair sign-flip permutation p-value.

Confirmatory target (protocol Section 1): per-token arm, base model, pooled
natural-language pairs, dominance within {2,3,4}. The Holm-Bonferroni family
is exactly that cell's in-window dimensions. Everything else (the Instruct
replication, the attention arm, per-type robustness, H0/H1) is reported as
secondary and never enters the confirmatory correction.

Normalization (protocol Section 4.2): the per-token arm divides lifetimes by
the item's point-cloud diameter. If the topology stage did not record a
diameter, this script runs unnormalized and records normalization="none" so
the result is not mistaken for normalized. The attention arm is on a [0,1]
density filtration and needs no scale.

Output: <analysis.output_dir>/results.json. Figures are produced separately
by src/analysis/figures.py.

Cost: $0, runs locally on the topology outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.bootstrap import (
    bootstrap_summary,
    max_persistence,
    paired_bootstrap,
    paired_cohens_dz,
)
from src.analysis.dominance import dimensional_dominance_test, holm_bonferroni_correction
from src.analysis.permutation import matched_pair_permutation_test

# Preregistered constants (docs/analysis_protocol.md Sections 4-5). Fixed here,
# not config-tunable, so the preregistered thresholds cannot drift per run.
MARGIN = 0.2
FLOOR = 0.5
ALPHA = 0.05
SEARCH_WINDOW = (2, 3, 4)
MAX_DIM = 4
PRIMARY_TYPE = "pooled_natural_language"


def _log(msg: str) -> None:
    print(msg)


def load_pairs(pairs_path: Path) -> list[dict]:
    """Read pairs.jsonl into a list of pairing records."""
    pairs = []
    with Path(pairs_path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def per_token_scale(npz) -> tuple[float, bool]:
    """Return (scale, normalized). Uses a 'diameter' key if the topology stage
    recorded one; otherwise scale 1.0 and normalized=False."""
    if "diameter" in npz.files:
        d = float(np.asarray(npz["diameter"]).reshape(-1)[0])
        if d > 0:
            return d, True
    return 1.0, False


def per_token_mk(npz_path: Path, dim: int) -> tuple[float, bool]:
    """Max persistence at dimension `dim` for a per-token diagram file."""
    with np.load(npz_path, allow_pickle=True) as npz:
        scale, normalized = per_token_scale(npz)
        key = f"dim_{dim}"
        dgm = npz[key] if key in npz.files else None
        return max_persistence(dgm, scale=scale), normalized


def attention_mk(npz_path: Path, dim: int) -> float:
    """Max persistence at `dim` for the attention arm, taken at the threshold
    whose density is closest to the median (representative-threshold convention).
    The attention filtration is already on [0,1], so scale is 1.0."""
    with np.load(npz_path, allow_pickle=True) as npz:
        if "densities" in npz.files and np.asarray(npz["densities"]).size > 0:
            densities = np.asarray(npz["densities"]).reshape(-1)
            idx = int(np.argmin(np.abs(densities - np.median(densities))))
        else:
            idx = 0
        key = f"thresh_{idx}_dim_{dim}"
        dgm = npz[key] if key in npz.files else None
        return max_persistence(dgm, scale=1.0)


def _resolve(topo_out: Path, rel_path: str | None) -> Path | None:
    return (topo_out / rel_path) if rel_path else None


def pair_mk_arrays(
    pairs: list[dict],
    items: dict,
    topo_out: Path,
    arm: str,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
    """Build pair-aligned M_k arrays for one (arm, dim) over a set of pairs.

    Returns (contradiction_mk, control_mk, kept_pair_ids, normalized).
    Pairs are skipped when either member's diagram for this arm is missing or
    errored. `normalized` is True only if every kept per-token item carried a
    diameter (attention arm reports normalized=True trivially).
    """
    path_key = "per_token_path" if arm == "per_token" else "attention_path"
    contra_vals: list[float] = []
    control_vals: list[float] = []
    kept: list[str] = []
    all_normalized = True

    for pr in pairs:
        cid = pr["contradiction_id"]
        kid = pr["control_id"]
        c_entry = items.get(cid)
        k_entry = items.get(kid)
        if c_entry is None or k_entry is None:
            continue
        c_path = _resolve(topo_out, c_entry.get(path_key))
        k_path = _resolve(topo_out, k_entry.get(path_key))
        if c_path is None or k_path is None or not c_path.exists() or not k_path.exists():
            continue
        if arm == "per_token":
            c_mk, c_norm = per_token_mk(c_path, dim)
            k_mk, k_norm = per_token_mk(k_path, dim)
            all_normalized = all_normalized and c_norm and k_norm
        else:
            c_mk = attention_mk(c_path, dim)
            k_mk = attention_mk(k_path, dim)
        contra_vals.append(c_mk)
        control_vals.append(k_mk)
        kept.append(pr["pair_id"])

    return (
        np.asarray(contra_vals, dtype=np.float64),
        np.asarray(control_vals, dtype=np.float64),
        kept,
        all_normalized if arm == "per_token" else True,
    )


def cell_dim_stats(
    contra: np.ndarray,
    control: np.ndarray,
    boot_iters: int,
    perm_iters: int,
    conf_level: float,
) -> dict:
    """Effect size, bootstrap CI, and permutation p for one (arm, model, type, dim)."""
    n = int(contra.size)
    if n < 2:
        return {
            "d_z": 0.0,
            "ci": [0.0, 0.0],
            "p_value": 1.0,
            "n_pairs": n,
            "testable": False,
        }
    d_z = paired_cohens_dz(contra, control)
    boot = paired_bootstrap(contra, control, n_iterations=boot_iters)
    summ = bootstrap_summary(boot, confidence_level=conf_level)
    perm = matched_pair_permutation_test(contra, control, n_iterations=perm_iters)
    return {
        "d_z": float(d_z),
        "ci": [summ["lower_ci"], summ["upper_ci"]],
        "p_value": float(perm["p_value"]),
        "n_pairs": n,
        "testable": True,
    }


def analyze_cell(
    pairs: list[dict],
    items: dict,
    topo_out: Path,
    arm: str,
    boot_iters: int,
    perm_iters: int,
    conf_level: float,
) -> dict:
    """Run all dimensions 0..MAX_DIM for one (arm, model, type) cell."""
    per_dim: dict[int, dict] = {}
    effect_sizes: dict[int, float] = {}
    p_values: dict[int, float] = {}
    normalized_flags: list[bool] = []
    n_pairs_seen = 0

    for k in range(MAX_DIM + 1):
        contra, control, kept, normalized = pair_mk_arrays(pairs, items, topo_out, arm, k)
        normalized_flags.append(normalized)
        n_pairs_seen = max(n_pairs_seen, len(kept))
        stats = cell_dim_stats(contra, control, boot_iters, perm_iters, conf_level)
        per_dim[k] = stats
        effect_sizes[k] = stats["d_z"]
        p_values[k] = stats["p_value"]

    dominance = dimensional_dominance_test(
        effect_sizes, p_values, margin=MARGIN, floor=FLOOR, search_window=SEARCH_WINDOW
    )

    return {
        "arm": arm,
        "n_pairs": n_pairs_seen,
        "normalized": all(normalized_flags),
        "per_dim": {str(k): per_dim[k] for k in per_dim},
        "dominance": dominance,
    }


def analyze(cfg: dict, log=_log) -> dict:
    """Full analysis. Returns the results dict (also what main writes to JSON)."""
    corpus_dir = Path(cfg["corpus"]["output_dir"])
    topo_out = Path(cfg["topology"]["output_dir"])
    boot_iters = int(cfg["analysis"]["bootstrap"]["n_iterations"])
    conf_level = float(cfg["analysis"]["bootstrap"].get("confidence_level", 0.95))
    perm_iters = int(cfg["analysis"].get("permutation", {}).get("n_iterations", 10000))

    pairs = load_pairs(corpus_dir / "pairs.jsonl")
    log(f"Loaded {len(pairs)} matched pairs from {corpus_dir / 'pairs.jsonl'}")

    with (topo_out / "topology_manifest.json").open() as f:
        manifest = json.load(f)

    # Identify models by role.
    models = manifest["models"]
    base_name = None
    for name, entry in models.items():
        if entry.get("role") == "base":
            base_name = name
            break
    if base_name is None:
        base_name = next(iter(models))  # fall back to the only/first model
        log(f"No model with role 'base' found; using {base_name}")

    # Pairs grouped by contradiction type, plus the pooled natural-language set.
    def pairs_of_type(t):
        return [p for p in pairs if p.get("contradiction_type") == t]

    types_present = sorted({p.get("contradiction_type") for p in pairs if p.get("contradiction_type")})

    results: dict = {
        "config_name": cfg.get("experiment", {}).get("name", "unknown"),
        "protocol": {
            "primary_statistic": "max_persistence_paired_dz",
            "margin": MARGIN,
            "floor": FLOOR,
            "alpha": ALPHA,
            "search_window": list(SEARCH_WINDOW),
        },
        "iterations": {"bootstrap": boot_iters, "permutation": perm_iters},
        "primary": {},
        "secondary": {},
    }

    # ---- Primary confirmatory cell: per-token, base, pooled NL ----
    base_items = models[base_name]["items"]
    primary = analyze_cell(pairs, base_items, topo_out, "per_token",
                           boot_iters, perm_iters, conf_level)
    primary["cell"] = {"arm": "per_token", "model": base_name, "type": PRIMARY_TYPE}
    primary["normalization"] = "diameter" if primary["normalized"] else "none"

    # Holm-Bonferroni over the in-window dims of the primary cell only.
    window_p = {k: primary["per_dim"][str(k)]["p_value"] for k in SEARCH_WINDOW}
    holm = holm_bonferroni_correction(window_p, alpha=ALPHA)
    primary["holm"] = {
        "family_dims": list(SEARCH_WINDOW),
        "adjusted_p_values": {str(k): holm["adjusted_p_values"][k] for k in window_p},
        "rejected": {str(k): holm["rejected"][k] for k in window_p},
    }

    best = primary["dominance"]["best_in_search_window"]
    kstar = best["dim"]
    kstar_significant = bool(holm["rejected"].get(kstar, False))
    primary["confirmatory"] = {
        "k_star": kstar,
        "is_dominant_by_effect": best["is_dominant"],
        "k_star_significant_holm": kstar_significant,
        "confirmed": bool(best["is_dominant"] and kstar_significant),
    }
    results["primary"] = primary

    if primary["normalization"] == "none":
        log("WARNING: per-token arm ran UNNORMALIZED (topology stage recorded no "
            "diameter). Primary result is provisional until 03 records diameters.")

    # ---- Secondary cells ----
    def add_secondary(key, cell):
        results["secondary"][key] = cell

    # Attention arm, base, pooled NL.
    if any(it.get("attention_path") for it in base_items.values()):
        att = analyze_cell(pairs, base_items, topo_out, "attention",
                           boot_iters, perm_iters, conf_level)
        att["cell"] = {"arm": "attention", "model": base_name, "type": PRIMARY_TYPE}
        add_secondary(f"attention__{base_name}__{PRIMARY_TYPE}", att)

    # Instruct replication, per-token, pooled NL.
    for name, entry in models.items():
        if entry.get("role") == "instruct":
            rep = analyze_cell(pairs, entry["items"], topo_out, "per_token",
                               boot_iters, perm_iters, conf_level)
            rep["cell"] = {"arm": "per_token", "model": name, "type": PRIMARY_TYPE}
            rep["normalization"] = "diameter" if rep["normalized"] else "none"
            add_secondary(f"per_token__{name}__{PRIMARY_TYPE}", rep)

    # Per-type robustness, per-token, base.
    for t in types_present:
        tp = pairs_of_type(t)
        if len(tp) >= 2:
            cell = analyze_cell(tp, base_items, topo_out, "per_token",
                                boot_iters, perm_iters, conf_level)
            cell["cell"] = {"arm": "per_token", "model": base_name, "type": t}
            add_secondary(f"per_token__{base_name}__{t}", cell)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    args = parser.parse_args()

    from src.utils.config import ensure_dir, load_config
    from src.utils.logging import get_logger

    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg["analysis"]["output_dir"])
    log = get_logger("analyze", log_dir=out_dir / "logs")

    results = analyze(cfg, log=log.info)

    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Wrote results to {out_path}")

    conf = results["primary"]["confirmatory"]
    log.info(f"Primary confirmatory: k*={conf['k_star']}, "
             f"dominant_by_effect={conf['is_dominant_by_effect']}, "
             f"significant={conf['k_star_significant_holm']}, "
             f"CONFIRMED={conf['confirmed']}")


if __name__ == "__main__":
    main()
