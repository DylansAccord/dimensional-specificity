#!/usr/bin/env python
"""
End-to-end smoke test: 01 -> 02 -> 03 -> 04 on a tiny synthetic corpus.

The corpus (01) and inference (02) stages need Hugging Face downloads and a
GPU respectively, so this harness MOCKS their outputs on CPU: it writes
synthetic per-token embeddings and attention matrices plus a 02-style
inference manifest, and a 01-style pairs.jsonl. It then runs the REAL
scripts/03_topology.py and scripts/04_analyze.py as subprocesses and checks
the whole chain produces a coherent results.json.

What it actually proves:
  - 03 consumes the 02 manifest contract, runs both arms, and writes per-token
    diagrams that now carry a 'diameter' (the protocol Section 4.2 normalization
    scale recorded by 03).
  - 04 consumes pairs.jsonl + the topology manifest, forms matched pairs,
    reports normalization == "diameter" (the end-to-end diameter round-trip),
    and emits a well-formed results.json.

The synthetic geometry plants a soft per-token difference (contradiction items
sit on a sphere shell, controls are a blob) so the run also exercises real
topology on a real signal, but the pass/fail gate is the plumbing, not the
detection: small-n topology is noisy and the statistics are validated
separately by the module sanity checks and the 04 validator.

Run: python scripts/_smoke_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
MODEL = "smoke-base"
D = 8          # embedding dimension (small for speed)
N_TOK = 14     # tokens per item
N_LAYERS = 4   # attention layers


def _sphere_shell(n, rng):
    """n points on a noisy unit sphere shell in R^3, padded to R^D. Has an H2 void."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v += rng.normal(0, 0.05, size=(n, 3))
    pad = rng.normal(0, 0.05, size=(n, D - 3))
    return np.hstack([v, pad]).astype(np.float64)


def _blob(n, rng):
    """n points in a Gaussian blob in R^D. No higher-dim void."""
    return rng.normal(0, 0.4, size=(n, D)).astype(np.float64)


def _attention(n_tok, n_layers, dense, rng):
    """n_layers attention matrices (n_tok, n_tok). If dense, plant a clique block."""
    mats = []
    for _ in range(n_layers):
        A = np.abs(rng.normal(0.1, 0.02, size=(n_tok, n_tok)))
        if dense:
            blk = slice(0, n_tok // 2)
            A[blk, blk] += 0.8
        A = (A + A.T) / 2.0
        np.fill_diagonal(A, 0.0)
        mats.append(A.astype(np.float64))
    return mats


def build_smoke_inputs(root: Path, n_pairs: int = 6, seed: int = 0) -> Path:
    """Write mock 02 + 01 outputs and a smoke config. Returns the config path."""
    rng = np.random.default_rng(seed)
    inf_dir = root / "inference"
    model_dir = inf_dir / MODEL
    corpus_dir = root / "corpus"
    for d in (model_dir, corpus_dir):
        d.mkdir(parents=True, exist_ok=True)

    items: dict = {}
    pairs: list[dict] = []

    def write_item(item_id, label, cls, embeddings, attention_mats):
        emb_path = model_dir / f"{item_id}__embeddings.npz"
        att_path = model_dir / f"{item_id}__attention.npz"
        np.savez_compressed(emb_path, embeddings=embeddings)
        att_dict = {f"layer_{li}": A for li, A in enumerate(attention_mats)}
        np.savez_compressed(att_path, **att_dict)
        items[item_id] = {
            "id": item_id,
            "label": label,
            "source": "smoke",
            "class": cls,
            "embedding_path": str(emb_path.relative_to(inf_dir)),
            "attention_path": str(att_path.relative_to(inf_dir)),
            "error": None,
        }

    for i in range(n_pairs):
        cid, kid = f"con_{i}", f"ctrl_{i}"
        write_item(cid, "contradiction", "contradiction",
                   _sphere_shell(N_TOK, rng), _attention(N_TOK, N_LAYERS, True, rng))
        write_item(kid, "neutral", "baseline",
                   _blob(N_TOK, rng), _attention(N_TOK, N_LAYERS, False, rng))
        pairs.append({"pair_id": f"pair_{cid}", "contradiction_id": cid,
                      "control_id": kid, "contradiction_type": None, "match_score": 0.5})

    inf_manifest = {
        "config_name": "smoke",
        "models": {MODEL: {"name": MODEL, "role": "base",
                           "quantization": "none", "items": items}},
    }
    with (inf_dir / "manifest.json").open("w") as f:
        json.dump(inf_manifest, f)
    with (corpus_dir / "pairs.jsonl").open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    cfg = {
        "experiment": {"name": "smoke", "random_seed": 0},
        "huggingface": {"token": "hf_smoke_placeholder", "cache_dir": str(root / "hf_cache")},
        "corpus": {"output_dir": str(corpus_dir)},
        "inference": {"output_dir": str(inf_dir)},
        "topology": {
            "output_dir": str(root / "topology"),
            "sentence_embedding_arm": {"max_dim": 4, "sparse_filtration_threshold": 500},
            "attention_graph_arm": {"max_dim": 4, "edge_weight_thresholds": 3},
        },
        "analysis": {
            "output_dir": str(root / "analysis"),
            "bootstrap": {"n_iterations": 200, "confidence_level": 0.95},
            "permutation": {"n_iterations": 500},
        },
    }
    cfg_path = root / "smoke.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path


def _run_script(script: str, cfg_path: Path):
    return subprocess.run(
        [sys.executable, script, "--config", str(cfg_path)],
        cwd=str(REPO), capture_output=True, text=True,
    )


def run_smoke() -> int:
    failures: list[str] = []

    def check(name, cond, detail=""):
        marker = "PASS" if cond else "FAIL"
        tail = f": {detail}" if (detail and not cond) else ""
        print(f"  [{marker}] {name}{tail}")
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg_path = build_smoke_inputs(root, n_pairs=6, seed=0)
        print(f"  built synthetic 01/02 outputs in {root}")

        # ---- run real 03 ----
        r3 = _run_script("scripts/03_topology.py", cfg_path)
        check("03 exits 0", r3.returncode == 0,
              detail=f"rc={r3.returncode}\n{r3.stdout[-800:]}\n{r3.stderr[-800:]}")
        topo_manifest = root / "topology" / "topology_manifest.json"
        check("03 wrote topology_manifest.json", topo_manifest.exists())

        if topo_manifest.exists():
            man = json.loads(topo_manifest.read_text())
            base_items = man["models"][MODEL]["items"]
            check("03 role propagated as base", man["models"][MODEL].get("role") == "base")
            n_err = sum(1 for it in base_items.values() if it.get("error"))
            check("03 produced no per-item errors", n_err == 0, detail=f"{n_err} errors")
            # diameter round-trip: a per_token npz carries a positive scalar diameter.
            some = next((it for it in base_items.values() if it.get("per_token_path")), None)
            check("03 per_token output present", some is not None)
            if some:
                z = np.load(root / "topology" / some["per_token_path"])
                check("03 per_token npz carries a diameter", "diameter" in z.files)
                if "diameter" in z.files:
                    check("03 diameter is a positive scalar",
                          z["diameter"].ndim == 0 and float(z["diameter"]) > 0,
                          detail=f"{z['diameter']}")

        # ---- run real 04 ----
        r4 = _run_script("scripts/04_analyze.py", cfg_path)
        check("04 exits 0", r4.returncode == 0,
              detail=f"rc={r4.returncode}\n{r4.stdout[-800:]}\n{r4.stderr[-800:]}")
        results_path = root / "analysis" / "results.json"
        check("04 wrote results.json", results_path.exists())

        if results_path.exists():
            res = json.loads(results_path.read_text())
            prim = res.get("primary", {})
            check("04 results has a primary cell", bool(prim.get("per_dim")))
            check("04 primary covers dims 0..4",
                  set(prim.get("per_dim", {}).keys()) == {"0", "1", "2", "3", "4"},
                  detail=f"{sorted(prim.get('per_dim', {}).keys())}")
            check("04 end-to-end diameter round-trip (normalization == diameter)",
                  prim.get("normalization") == "diameter",
                  detail=f"got {prim.get('normalization')}")
            check("04 confirmatory block present",
                  "confirmatory" in prim and "confirmed" in prim["confirmatory"])
            check("04 n_pairs == 6", prim.get("n_pairs") == 6, detail=str(prim.get("n_pairs")))
            # informational only: did the planted geometry move H2?
            dz2 = prim.get("per_dim", {}).get("2", {}).get("d_z")
            print(f"  [INFO] primary H2 d_z = {dz2} (planted signal; not a gate)")

    print()
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("SMOKE TEST PASSED: 01->02(mock)->03->04 chained end to end.")
    return 0


if __name__ == "__main__":
    print("end-to-end smoke test\n")
    sys.exit(run_smoke())
