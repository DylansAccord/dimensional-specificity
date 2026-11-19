"""
Topology driver: compute persistent homology diagrams for each item.

For each item from scripts/02_inference.py's output manifest:
  - Per-token arm: Vietoris-Rips persistence on the contextual embedding
    point cloud, dimensions 0 through max_dim. Uses src/topology/vietoris_rips.py.
  - Attention arm: flag-complex + zigzag persistence over the layer sequence,
    swept across n_thresholds density-equivalent threshold values. Uses
    src/topology/flag_complex.py and src/topology/zigzag.py.

Both arms compute H_0 through H_max_dim per the H_1-as-candidate refinement
in the handoff: H_1 is a candidate dimension on equal footing with H_2-H_4,
not a control. The analysis pipeline (scripts/04) is where dimensional
exclusion arguments are applied; this script is dimension-neutral.

Attention-graph construction per layer:
  - If 3D (n_heads, n_tokens, n_tokens), mean over heads.
  - Symmetrize: A_sym = (A + A.T) / 2.
  - Zero the diagonal (self-attention is uninformative).
  - Density-equivalent thresholding: keep the top fraction of edges by weight.

Flag complex simplex dimension is set to max_dim + 1 because computing H_k
requires (k+1)-simplices to bound the k-cycles.

Outputs:
  - topology_manifest.json at topology.output_dir/topology_manifest.json
  - per-item diagram files under topology.output_dir/<model_safename>/
      <item_id>__per_token.npz: keys 'dim_0', 'dim_1', ..., 'dim_max_dim'
      <item_id>__attention.npz: keys 'densities', 'layers',
                                'thresh_<i>_dim_<k>' for each (i, k)

Usage:
    python scripts/03_topology.py --config configs/dry_run.yaml

Local validation:
    python scripts/_validate_03_schema.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist

# Make src importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topology.flag_complex import build_flag_complex
from src.topology.vietoris_rips import compute_vr_persistence
from src.topology.zigzag import compute_zigzag_persistence
from src.utils.config import ensure_dir, load_config

logger = logging.getLogger("murmur.topology")

DEFAULT_DENSITY_RANGE = (0.05, 0.50)


# ---- Per-token arm ----


def compute_per_token_topology(
    embedding_path: Path,
    max_dim: int,
    sparse_threshold: int,
) -> dict[int, np.ndarray]:
    """Compute VR persistence on a per-token embedding point cloud.

    Returns dict mapping dimension k -> diagram array of shape (n_features, 2).
    Death column may contain np.inf for essential classes.
    """
    npz = np.load(embedding_path, allow_pickle=True)
    embeddings = npz["embeddings"]
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, got shape {embeddings.shape}"
        )
    result = compute_vr_persistence(
        embeddings,
        max_dim=max_dim,
        sparse_threshold=sparse_threshold,
    )
    return {k: result.diagrams[k] for k in range(min(len(result.diagrams), max_dim + 1))}


def point_cloud_diameter(embeddings: np.ndarray) -> float:
    """Max pairwise Euclidean distance in a point cloud.

    This is the per-item normalization scale for the per-token arm (protocol
    Section 4.2). Returns 0.0 for fewer than two points; the analysis stage
    treats a non-positive scale as "no normalization available".
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return 0.0
    return float(pdist(X).max())


# ---- Attention arm ----


def threshold_attention_to_graph(
    attention_matrix: np.ndarray,
    density: float,
) -> nx.Graph:
    """Build an undirected weighted graph from an attention matrix.

    Mean over heads if 3D, symmetrize, zero diagonal, then keep the top
    `density` fraction of off-diagonal edges by weight.

    Args:
        attention_matrix: shape (n_heads, n_tokens, n_tokens) or (n_tokens, n_tokens).
        density: fraction of edges to retain, in (0, 1].

    Returns:
        networkx.Graph. Nodes are token indices 0..n_tokens-1. Edges carry
        the 'weight' attribute (symmetric attention value).
    """
    if attention_matrix.ndim == 3:
        A = attention_matrix.mean(axis=0)
    elif attention_matrix.ndim == 2:
        A = attention_matrix
    else:
        raise ValueError(
            f"attention_matrix must be 2D or 3D, got shape {attention_matrix.shape}"
        )
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"attention matrix must be square, got {A.shape}")

    A_sym = (A + A.T) / 2.0
    np.fill_diagonal(A_sym, 0.0)

    iu = np.triu_indices(n, k=1)
    weights = A_sym[iu]

    g = nx.Graph()
    g.add_nodes_from(range(n))

    if weights.size == 0:
        return g

    n_keep = max(1, int(round(density * weights.size)))
    if n_keep >= weights.size:
        threshold = -np.inf
    else:
        threshold = np.partition(weights, -n_keep)[-n_keep]

    rows, cols = iu
    keep_mask = weights >= threshold
    for r, c in zip(rows[keep_mask], cols[keep_mask]):
        g.add_edge(int(r), int(c), weight=float(A_sym[r, c]))

    return g


def compute_attention_topology(
    attention_path: Path,
    max_dim: int,
    n_thresholds: int,
    density_range: tuple[float, float] = DEFAULT_DENSITY_RANGE,
) -> dict:
    """Compute zigzag persistence on attention-graph flag complexes.

    For each of n_thresholds density values evenly spaced across density_range,
    build a sequence of flag complexes (one per layer), compute zigzag
    persistence, and store diagrams per dimension.

    Returns dict:
      - 'densities': np.ndarray of threshold densities used
      - 'layers': np.ndarray of layer indices in sequence order
      - 'diagrams': {threshold_idx: {dim: np.ndarray (n_features, 2)}}
    """
    npz = np.load(attention_path, allow_pickle=True)
    layer_keys = sorted(
        (k for k in npz.files if k.startswith("layer_")),
        key=lambda k: int(k.split("_")[1]),
    )
    if not layer_keys:
        raise ValueError(f"No layer_* arrays in {attention_path}")

    layer_indices = [int(k.split("_")[1]) for k in layer_keys]
    attention_per_layer = [npz[k] for k in layer_keys]

    densities = np.linspace(density_range[0], density_range[1], n_thresholds)
    flag_simplex_dim = max_dim + 1  # need (k+1)-simplices for H_k

    diagrams_by_threshold: dict[int, dict[int, np.ndarray]] = {}
    for ti, density in enumerate(densities):
        complex_seq = []
        for A in attention_per_layer:
            g = threshold_attention_to_graph(A, density=float(density))
            complex_i = build_flag_complex(g, max_dim=flag_simplex_dim)
            complex_seq.append(complex_i)
        dgms = compute_zigzag_persistence(complex_seq, max_dim=max_dim)
        diagrams_by_threshold[ti] = {
            dim: _intervals_to_array(dgms.get(dim, []))
            for dim in range(max_dim + 1)
        }

    return {
        "densities": densities,
        "layers": np.array(layer_indices),
        "diagrams": diagrams_by_threshold,
    }


def _intervals_to_array(intervals: list[tuple[float, float]]) -> np.ndarray:
    """Convert list of (birth, death) tuples to a (n, 2) numpy array."""
    if not intervals:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(intervals, dtype=np.float64)


# ---- File I/O ----


def save_per_token_diagrams(
    diagrams: dict[int, np.ndarray],
    out_path: Path,
    diameter: float | None = None,
) -> None:
    """Save per-token diagrams as npz with keys 'dim_0', 'dim_1', ...

    If diameter is given, it is stored under the 'diameter' key so the analysis
    stage can normalize lifetimes by the point-cloud diameter (protocol Section
    4.2). Omitting it preserves the older diagram-only format.
    """
    save_dict = {f"dim_{k}": v for k, v in diagrams.items()}
    if diameter is not None:
        save_dict["diameter"] = np.array(float(diameter), dtype=np.float64)
    np.savez_compressed(out_path, **save_dict)


def save_attention_diagrams(result: dict, out_path: Path) -> None:
    """Save attention diagrams as npz.

    Keys:
      - 'densities': array of threshold density values
      - 'layers': array of layer indices in sequence order
      - 'thresh_<i>_dim_<k>': diagram at threshold index i, dimension k
    """
    save_dict: dict = {
        "densities": result["densities"],
        "layers": result["layers"],
    }
    for ti, dgms in result["diagrams"].items():
        for dim, dgm in dgms.items():
            save_dict[f"thresh_{ti}_dim_{dim}"] = dgm
    np.savez_compressed(out_path, **save_dict)


# ---- Main orchestration ----


def main(config_path: str) -> int:
    cfg = load_config(config_path)

    inference_output_dir = Path(cfg["inference"]["output_dir"])
    inference_manifest_path = inference_output_dir / "manifest.json"
    if not inference_manifest_path.exists():
        logger.error(
            "Inference manifest not found at %s. Run scripts/02_inference.py first.",
            inference_manifest_path,
        )
        return 1
    with inference_manifest_path.open() as f:
        inf_manifest = json.load(f)
    logger.info("Loaded inference manifest from %s", inference_manifest_path)

    topo_cfg = cfg["topology"]
    topo_out = ensure_dir(Path(topo_cfg["output_dir"]))

    pt_cfg = topo_cfg.get("sentence_embedding_arm", {})
    pt_max_dim = int(pt_cfg.get("max_dim", 4))
    pt_sparse_thresh = int(pt_cfg.get("sparse_filtration_threshold", 500))

    att_cfg = topo_cfg.get("attention_graph_arm", {})
    att_max_dim = int(att_cfg.get("max_dim", 4))
    att_n_thresholds = int(att_cfg.get("edge_weight_thresholds", 10))

    topology_manifest: dict = {
        "config_name": cfg["experiment"]["name"],
        "inference_manifest": str(inference_manifest_path),
        "output_dir": str(topo_cfg["output_dir"]),
        "per_token_arm": {
            "max_dim": pt_max_dim,
            "sparse_threshold": pt_sparse_thresh,
        },
        "attention_arm": {
            "max_dim": att_max_dim,
            "n_thresholds": att_n_thresholds,
            "density_range": list(DEFAULT_DENSITY_RANGE),
        },
        "models": {},
    }

    for model_name, model_entry in inf_manifest["models"].items():
        model_safename = model_name.replace("/", "__")
        model_topo_dir = ensure_dir(topo_out / model_safename)
        items_topo: dict = {}

        logger.info("Processing model %s", model_name)
        items = model_entry.get("items", {})
        n_total = len(items)

        for i, (item_id, item_entry) in enumerate(items.items()):
            entry: dict = {
                "id": item_id,
                "label": item_entry.get("label"),
                "class": item_entry.get("class"),
                "per_token_path": None,
                "attention_path": None,
            }

            try:
                if item_entry.get("embedding_path"):
                    emb_full = inference_output_dir / item_entry["embedding_path"]
                    pt_diagrams = compute_per_token_topology(
                        emb_full,
                        max_dim=pt_max_dim,
                        sparse_threshold=pt_sparse_thresh,
                    )
                    emb_cloud = np.load(emb_full, allow_pickle=True)["embeddings"]
                    pt_diameter = point_cloud_diameter(emb_cloud)
                    pt_out = model_topo_dir / f"{item_id}__per_token.npz"
                    save_per_token_diagrams(pt_diagrams, pt_out, diameter=pt_diameter)
                    entry["per_token_path"] = str(pt_out.relative_to(topo_out))

                if item_entry.get("attention_path"):
                    att_full = inference_output_dir / item_entry["attention_path"]
                    att_result = compute_attention_topology(
                        att_full,
                        max_dim=att_max_dim,
                        n_thresholds=att_n_thresholds,
                    )
                    att_out = model_topo_dir / f"{item_id}__attention.npz"
                    save_attention_diagrams(att_result, att_out)
                    entry["attention_path"] = str(att_out.relative_to(topo_out))
            except Exception as e:
                logger.exception("Failed on item %s: %s", item_id, e)
                entry["error"] = str(e)

            items_topo[item_id] = entry

            if (i + 1) % 10 == 0 or (i + 1) == n_total:
                logger.info("  progress: %d / %d items", i + 1, n_total)

        topology_manifest["models"][model_name] = {
            "name": model_name,
            "role": model_entry.get("role"),
            "items": items_topo,
        }

    out_manifest = topo_out / "topology_manifest.json"
    with out_manifest.open("w") as f:
        json.dump(topology_manifest, f, indent=2)
    logger.info("Wrote topology manifest: %s", out_manifest)

    return 0


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


if __name__ == "__main__":
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Topology driver: VR + zigzag persistence per item.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    sys.exit(main(args.config))
