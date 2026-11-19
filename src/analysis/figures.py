"""
Figures for the dimensional-specificity experiment.

Reads what scripts/04_analyze.py produces (a results dict) plus the topology
stage's per-item diagrams, and writes PNG figures. Two figure types, both
derived without recomputation so they can never silently disagree with the
statistics in results.json:

  1. Dominance sweep (per cell): paired Cohen's d_z by homology dimension with
     bootstrap CI error bars, the search window {2,3,4} shaded, the floor line
     at d_z = 0.5, and the winning dimension highlighted. This is the primary
     visualization of the dimensional-specificity claim.

  2. Persistence diagram (per example item): birth-death scatter per dimension
     from a per_token .npz, for visual QA of the topology stage.

The retired bottleneck-distance histogram is intentionally absent: under the
max-persistence statistic the dominance sweep already shows the standardized
separation per dimension. Bottleneck distance remains in
src/topology/distances.py for ad hoc diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display; write files only

import matplotlib.pyplot as plt
import numpy as np

FLOOR = 0.5
SEARCH_WINDOW = (2, 3, 4)
_DIM_COLOR = {0: "#34495e", 1: "#2980b9", 2: "#c0392b", 3: "#27ae60", 4: "#8e44ad"}
_DIM_MARKER = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v"}


def plot_dominance_sweep(cell: dict, out_path: Path, title: str | None = None) -> Path:
    """Plot d_z by dimension with CI error bars for one analysis cell."""
    per_dim = cell["per_dim"]
    dims = sorted(int(k) for k in per_dim)
    dz = [per_dim[str(k)]["d_z"] for k in dims]
    lo = [per_dim[str(k)]["ci"][0] for k in dims]
    hi = [per_dim[str(k)]["ci"][1] for k in dims]
    yerr = [
        [max(0.0, dz[i] - lo[i]) for i in range(len(dims))],
        [max(0.0, hi[i] - dz[i]) for i in range(len(dims))],
    ]
    kstar = cell.get("dominance", {}).get("best_in_search_window", {}).get("dim")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvspan(min(SEARCH_WINDOW) - 0.4, max(SEARCH_WINDOW) + 0.4,
               color="#ecf0f1", zorder=0, label="search window")
    ax.axhline(0.0, color="#bdc3c7", lw=0.8, zorder=1)
    ax.axhline(FLOOR, ls="--", color="#7f8c8d", lw=1.0, zorder=1, label=f"floor d_z={FLOOR}")
    ax.errorbar(dims, dz, yerr=yerr, fmt="none", ecolor="#95a5a6", capsize=4, zorder=2)
    colors = ["#c0392b" if k == kstar else "#34495e" for k in dims]
    ax.scatter(dims, dz, c=colors, s=70, zorder=3)
    ax.set_xticks(dims)
    ax.set_xlabel("homology dimension k")
    ax.set_ylabel("paired Cohen's d_z (contradiction - control)")
    ax.set_title(title or "dimensional dominance")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def plot_persistence_diagram(
    diagrams_by_dim: dict[int, np.ndarray],
    out_path: Path,
    title: str | None = None,
) -> Path:
    """Birth-death scatter for one item, colored by homology dimension."""
    fig, ax = plt.subplots(figsize=(5, 5))
    maxval = 1.0
    any_pts = False
    for k, dgm in sorted(diagrams_by_dim.items()):
        if dgm is None:
            continue
        arr = np.asarray(dgm, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] == 0:
            continue
        finite = np.isfinite(arr[:, 1])
        b, d = arr[finite, 0], arr[finite, 1]
        if d.size:
            maxval = max(maxval, float(d.max()))
            any_pts = True
        ax.scatter(b, d, s=30, marker=_DIM_MARKER.get(k, "o"),
                   c=_DIM_COLOR.get(k, "#000000"), alpha=0.8, label=f"H{k}")
    ax.plot([0, maxval], [0, maxval], color="#bdc3c7", lw=1.0, zorder=0)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title(title or "persistence diagram")
    if any_pts:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def load_diagrams(npz_path: Path, max_dim: int = 4) -> dict[int, np.ndarray]:
    """Load dim_0..dim_max_dim diagrams from a per_token .npz (ignores diameter)."""
    out: dict[int, np.ndarray] = {}
    with np.load(npz_path) as z:
        for k in range(max_dim + 1):
            key = f"dim_{k}"
            if key in z.files:
                out[k] = z[key]
    return out


def generate_all_figures(
    results: dict,
    topo_out: str | Path,
    fig_dir: str | Path,
    n_example_diagrams: int = 3,
) -> list[Path]:
    """Write all figures for a results dict. Returns the list of files written."""
    topo_out = Path(topo_out)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Dominance sweep for the primary cell and every secondary cell.
    prim = results.get("primary")
    if prim and prim.get("per_dim"):
        cell_desc = prim.get("cell", {})
        out = plot_dominance_sweep(prim, fig_dir / "dominance__primary.png",
                                   title=f"PRIMARY {cell_desc}")
        written.append(out)
    for key, cell in results.get("secondary", {}).items():
        if not cell.get("per_dim"):
            continue
        safe = key.replace("/", "_")
        out = plot_dominance_sweep(cell, fig_dir / f"dominance__{safe}.png", title=key)
        written.append(out)

    # A few example persistence diagrams from the topology manifest.
    man_path = topo_out / "topology_manifest.json"
    if man_path.exists():
        with man_path.open() as f:
            manifest = json.load(f)
        count = 0
        for _mname, mentry in manifest.get("models", {}).items():
            for iid, ie in mentry.get("items", {}).items():
                if count >= n_example_diagrams:
                    break
                rel = ie.get("per_token_path")
                if not rel:
                    continue
                npz_path = topo_out / rel
                if not npz_path.exists():
                    continue
                dgms = load_diagrams(npz_path)
                out = plot_persistence_diagram(dgms, fig_dir / f"pd__{iid}.png",
                                               title=f"{iid} (per-token)")
                written.append(out)
                count += 1
            if count >= n_example_diagrams:
                break

    return written


# ----- Sanity check (run: python -m src.analysis.figures) -----


def _sanity_check() -> None:
    import sys
    import tempfile

    failures: list[str] = []

    def check(name, cond, detail=""):
        marker = "PASS" if cond else "FAIL"
        tail = f": {detail}" if (detail and not cond) else ""
        print(f"  [{marker}] {name}{tail}")
        if not cond:
            failures.append(name)

    rng = np.random.default_rng(0)

    # Build a synthetic results dict shaped like scripts/04_analyze output, with
    # a clean dim-2 winner.
    def cell(winner=2):
        per_dim = {}
        for k in range(5):
            dz = 1.8 if k == winner else 0.1
            per_dim[str(k)] = {"d_z": dz, "ci": [dz - 0.2, dz + 0.2],
                               "p_value": 0.001 if k == winner else 0.4,
                               "n_pairs": 20, "testable": True}
        return {
            "cell": {"arm": "per_token", "model": "test-base", "type": "pooled_natural_language"},
            "per_dim": per_dim,
            "dominance": {"best_in_search_window": {"dim": winner, "is_dominant": True}},
            "normalization": "diameter",
        }

    results = {"primary": cell(2), "secondary": {"per_token__meta-llama/X__pooled": cell(3)}}

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        topo = td / "topology"
        model_dir = topo / "test-base"
        model_dir.mkdir(parents=True)
        # one per-token npz with a diameter key
        npz = model_dir / "it0__per_token.npz"
        np.savez_compressed(
            npz,
            dim_0=np.array([[0.0, 0.5], [0.0, 0.7]]),
            dim_1=np.array([[0.2, 0.9]]),
            dim_2=np.array([[0.3, 2.0]]),
            diameter=np.array(3.0),
        )
        manifest = {"models": {"test-base": {"name": "test-base", "role": "base",
                    "items": {"it0": {"id": "it0", "per_token_path": "test-base/it0__per_token.npz"}}}}}
        with (topo / "topology_manifest.json").open("w") as f:
            json.dump(manifest, f)

        fig_dir = td / "figures"
        written = generate_all_figures(results, topo, fig_dir)

        check("at least 3 figures written", len(written) >= 3, detail=f"got {len(written)}")
        check("all written files exist", all(p.exists() for p in written))
        check("all figures non-empty", all(p.stat().st_size > 0 for p in written))
        names = {p.name for p in written}
        check("primary dominance figure present", "dominance__primary.png" in names)
        check("secondary key sanitized (no slash in filename)",
              all("/" not in p.name for p in written))
        check("persistence diagram figure present",
              any(p.name.startswith("pd__") for p in written))

        # Direct single-figure calls also work.
        p1 = plot_dominance_sweep(results["primary"], fig_dir / "solo_dom.png", title="solo")
        check("solo dominance sweep written", p1.exists() and p1.stat().st_size > 0)
        p2 = plot_persistence_diagram(load_diagrams(npz), fig_dir / "solo_pd.png")
        check("solo persistence diagram written", p2.exists() and p2.stat().st_size > 0)

        # Empty-diagram robustness.
        p3 = plot_persistence_diagram({0: np.empty((0, 2)), 2: None}, fig_dir / "empty_pd.png")
        check("empty persistence diagram still writes a file", p3.exists())

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {failures}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    print("figures sanity check\n")
    _sanity_check()
