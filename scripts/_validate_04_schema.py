#!/usr/bin/env python
"""
Local schema + correctness validator for scripts/04_analyze.py.

Builds synthetic topology-stage outputs (a topology_manifest.json plus per-item
per_token .npz diagrams) and a synthetic pairs.jsonl in a temp directory, then
runs scripts/04_analyze.analyze on them. No GPU, no real model, no cloud.

It checks two scenarios:
  1. Planted signal: contradiction items carry a long-lived H_2 void, controls
     do not. The analysis must recover dim 2 as the dominant, significant,
     confirmed result.
  2. Null: contradiction and control diagrams are statistically identical. The
     analysis must NOT confirm dominance.

It also checks the normalization bookkeeping (diameter present vs absent).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import importlib.util

# Import 04 by file path (its name starts with a digit, so normal import won't work).
_spec = importlib.util.spec_from_file_location(
    "analyze04", str(Path(__file__).resolve().parent / "04_analyze.py")
)
analyze04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze04)


def _write_npz(path: Path, per_dim_lifetimes: dict[int, float], rng, with_diameter: bool):
    """Write a per_token .npz with one H_k feature per dim at the given lifetime
    (plus small noise so paired differences have non-zero variance)."""
    save = {}
    for k, life in per_dim_lifetimes.items():
        jitter = float(rng.normal(0, 0.02))
        birth = 0.0
        death = max(birth, life + jitter)
        save[f"dim_{k}"] = np.array([[birth, death]], dtype=np.float64)
    if with_diameter:
        save["diameter"] = np.array(1.0, dtype=np.float64)
    np.savez_compressed(path, **save)


def _build_dataset(root: Path, signal: bool, with_diameter: bool, n_pairs: int, seed: int):
    """Create corpus + topology outputs under root. Returns a cfg dict."""
    rng = np.random.default_rng(seed)
    corpus_dir = root / "corpus"
    topo_out = root / "topology"
    analysis_dir = root / "analysis"
    model_dir = topo_out / "test-base"
    for d in (corpus_dir, model_dir, analysis_dir):
        d.mkdir(parents=True, exist_ok=True)

    items = {}
    pairs = []
    for i in range(n_pairs):
        cid = f"con_{i}"
        kid = f"ctrl_{i}"
        # Contradiction: big H2 void when signal=True; small elsewhere.
        if signal:
            con_life = {0: 0.2, 1: 0.2, 2: 2.0, 3: 0.2, 4: 0.2}
        else:
            con_life = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
        ctrl_life = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}

        cpath = model_dir / f"{cid}__per_token.npz"
        kpath = model_dir / f"{kid}__per_token.npz"
        _write_npz(cpath, con_life, rng, with_diameter)
        _write_npz(kpath, ctrl_life, rng, with_diameter)

        rel_c = str(cpath.relative_to(topo_out))
        rel_k = str(kpath.relative_to(topo_out))
        items[cid] = {"id": cid, "label": "contradiction", "class": "contradiction",
                      "per_token_path": rel_c, "attention_path": None, "error": None}
        items[kid] = {"id": kid, "label": "neutral", "class": "baseline",
                      "per_token_path": rel_k, "attention_path": None, "error": None}
        pairs.append({"pair_id": f"pair_{cid}", "contradiction_id": cid,
                      "control_id": kid, "contradiction_type": None, "match_score": 0.5})

    manifest = {
        "config_name": "validate_04",
        "models": {"test-base": {"name": "test-base", "role": "base", "items": items}},
    }
    with (topo_out / "topology_manifest.json").open("w") as f:
        json.dump(manifest, f)
    with (corpus_dir / "pairs.jsonl").open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    return {
        "experiment": {"name": "validate_04"},
        "corpus": {"output_dir": str(corpus_dir)},
        "topology": {"output_dir": str(topo_out)},
        "analysis": {
            "output_dir": str(analysis_dir),
            "bootstrap": {"n_iterations": 200, "confidence_level": 0.95},
            "permutation": {"n_iterations": 500},
        },
    }


def _sanity_check():
    failures = []

    def check(name, cond, detail=""):
        marker = "PASS" if cond else "FAIL"
        tail = f": {detail}" if (detail and not cond) else ""
        print(f"  [{marker}] {name}{tail}")
        if not cond:
            failures.append(name)

    quiet = lambda *_a, **_k: None

    # --- Scenario 1: planted dim-2 signal, diameter present ---
    with tempfile.TemporaryDirectory() as td:
        cfg = _build_dataset(Path(td), signal=True, with_diameter=True, n_pairs=20, seed=1)
        res = analyze04.analyze(cfg, log=quiet)
        prim = res["primary"]
        conf = prim["confirmatory"]

        check("signal: k_star is dim 2", conf["k_star"] == 2, detail=f"got {conf['k_star']}")
        check("signal: dominant by effect", conf["is_dominant_by_effect"])
        check("signal: k_star significant under Holm", conf["k_star_significant_holm"])
        check("signal: CONFIRMED", conf["confirmed"])
        check("signal: normalization recorded as diameter",
              prim["normalization"] == "diameter", detail=prim["normalization"])
        dz2 = prim["per_dim"]["2"]["d_z"]
        dz3 = prim["per_dim"]["3"]["d_z"]
        check("signal: d_z(2) >> d_z(3)", dz2 > dz3 + 1.0, detail=f"dz2={dz2:.2f} dz3={dz3:.2f}")
        check("signal: n_pairs == 20", prim["n_pairs"] == 20, detail=str(prim["n_pairs"]))

    # --- Scenario 2: null, no signal ---
    with tempfile.TemporaryDirectory() as td:
        cfg = _build_dataset(Path(td), signal=False, with_diameter=True, n_pairs=20, seed=2)
        res = analyze04.analyze(cfg, log=quiet)
        conf = res["primary"]["confirmatory"]
        check("null: NOT confirmed", not conf["confirmed"],
              detail=f"confirmed={conf['confirmed']}")

    # --- Scenario 3: no diameter -> normalization 'none' ---
    with tempfile.TemporaryDirectory() as td:
        cfg = _build_dataset(Path(td), signal=True, with_diameter=False, n_pairs=12, seed=3)
        res = analyze04.analyze(cfg, log=quiet)
        check("no-diameter: normalization recorded as none",
              res["primary"]["normalization"] == "none",
              detail=res["primary"]["normalization"])
        # Signal should still be recoverable (un-normalized here, lifetimes are
        # already on a comparable scale in this synthetic data).
        check("no-diameter: still finds dim 2",
              res["primary"]["confirmatory"]["k_star"] == 2)

    # --- Scenario 4: results JSON is serializable and well-formed ---
    with tempfile.TemporaryDirectory() as td:
        cfg = _build_dataset(Path(td), signal=True, with_diameter=True, n_pairs=8, seed=4)
        res = analyze04.analyze(cfg, log=quiet)
        s = json.dumps(res)  # must not raise
        check("results JSON serializes", isinstance(s, str) and len(s) > 0)
        check("primary has expected top keys",
              {"cell", "per_dim", "dominance", "holm", "confirmatory", "normalization"}
              <= set(res["primary"].keys()))
        check("holm family is the in-window dims",
              res["primary"]["holm"]["family_dims"] == [2, 3, 4])

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {failures}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    print("04_analyze schema + correctness validation\n")
    _sanity_check()
