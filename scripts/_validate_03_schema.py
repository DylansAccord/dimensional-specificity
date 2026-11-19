"""
Local schema validator for scripts/03_topology.py.

This validator generates random point clouds and random attention matrices,
constructs a fake inference manifest pointing at them, runs scripts/03's
main() against it, and verifies that:
  - topology_manifest.json is well-formed
  - per-item .npz output files exist
  - persistence diagrams have the right keys and shapes

Unlike the 02 validator, this one runs REAL topology computation: real
ripser, real flag_complex, real zigzag. Only the upstream model (Llama) is
absent; we substitute random arrays of the right shape. Diagrams contain
whatever features random data produces, which is enough to validate the
pipeline machinery end-to-end.

No GPU required. Exits 0 on PASS, 1 on FAIL.

Usage:
    python scripts/_validate_03_schema.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _make_fake_inference_outputs(out_dir: Path) -> dict:
    """Write fake embedding + attention .npz files; return the manifest dict."""
    model_name = "fake/model"
    model_safe = "fake__model"
    model_dir = out_dir / model_safe
    model_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    items: dict = {}

    # 4 items: mix of has-attention and not, mix of classes.
    # 2 with attention exercises the attention arm; the other 2 exercise
    # the per-token arm only.
    item_specs = [
        ("t1", "contradiction", "contradiction", True),
        ("t2", "entailment", "baseline", True),
        ("t3", "contradiction", "synthetic", False),
        ("t4", "neutral", "baseline", False),
    ]
    n_tokens = 12
    hidden = 32   # small embedding dim for fast ripser
    n_heads = 2
    layer_range = [0, 4]

    for item_id, label, cls, has_attn in item_specs:
        emb = rng.standard_normal((n_tokens, hidden)).astype(np.float32)
        emb_path = model_dir / f"{item_id}__embeddings.npz"
        np.savez_compressed(
            emb_path,
            embeddings=emb,
            token_ids=np.arange(n_tokens),
            token_strings=np.array([f"tok{i}" for i in range(n_tokens)]),
            item_id=item_id,
        )

        entry = {
            "id": item_id,
            "label": label,
            "source": "fake",
            "class": cls,
            "embedding_path": f"{model_safe}/{item_id}__embeddings.npz",
            "attention_path": None,
        }

        if has_attn:
            att_path = model_dir / f"{item_id}__attention.npz"
            att_dict: dict = {}
            for layer_idx in range(layer_range[0], layer_range[1]):
                att = rng.random((n_heads, n_tokens, n_tokens)).astype(np.float32)
                att_dict[f"layer_{layer_idx}"] = att
            att_dict["token_ids"] = np.arange(n_tokens)
            att_dict["item_id"] = item_id
            np.savez_compressed(att_path, **att_dict)
            entry["attention_path"] = f"{model_safe}/{item_id}__attention.npz"

        items[item_id] = entry

    manifest = {
        "config_name": "schema_test",
        "config_path": "<test>",
        "corpus_dir": "<test>",
        "output_dir": str(out_dir),
        "n_items_total": len(items),
        "attention_layer_range": layer_range,
        "n_attention_subset": sum(1 for _, _, _, h in item_specs if h),
        "models": {
            model_name: {
                "name": model_name,
                "role": "base",
                "quantization": "4bit",
                "items": items,
            },
        },
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _make_fake_config(tmpdir: Path, inference_dir: Path, topology_dir: Path) -> Path:
    cfg = {
        "experiment": {"name": "schema_test_03", "random_seed": 42},
        "huggingface": {"token": "hf_fake"},
        "corpus": {"output_dir": str(tmpdir / "corpus_unused")},
        "inference": {
            "output_dir": str(inference_dir),
            "models": [{"name": "fake/model", "role": "base"}],
            "representations": {},
        },
        "topology": {
            "output_dir": str(topology_dir),
            "sentence_embedding_arm": {
                "library": "ripser",
                "max_dim": 2,        # small for validator speed
                "sparse_filtration_threshold": 500,
            },
            "attention_graph_arm": {
                "library": "dionysus",
                "max_dim": 2,        # small for validator speed
                "edge_weight_thresholds": 3,
            },
        },
        "analysis": {},
    }
    path = tmpdir / "schema_test_03.yaml"
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _load_script03():
    spec = importlib.util.spec_from_file_location(
        "_script03", REPO_ROOT / "scripts" / "03_topology.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        marker = "PASS" if cond else "FAIL"
        tail = f": {detail}" if (detail and not cond) else ""
        print(f"  [{marker}] {name}{tail}")
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        inference_dir = tmpdir / "embeddings"
        inference_dir.mkdir()
        topology_dir = tmpdir / "topology"
        # Don't create topology_dir; let the script create it.

        _make_fake_inference_outputs(inference_dir)
        cfg_path = _make_fake_config(tmpdir, inference_dir, topology_dir)

        script03 = _load_script03()
        rc = script03.main(str(cfg_path))
        check("main() returns 0", rc == 0, detail=f"got {rc}")

        manifest_path = topology_dir / "topology_manifest.json"
        check("topology_manifest.json exists", manifest_path.exists())
        if not manifest_path.exists():
            print("\nFAILED.")
            return 1

        with manifest_path.open() as f:
            topo_manifest = json.load(f)

        check("topology manifest has 'models'", "models" in topo_manifest)
        check(
            "topology manifest has 1 model",
            len(topo_manifest.get("models", {})) == 1,
        )

        model_entry = topo_manifest["models"].get("fake/model", {})
        check("fake/model present in topology manifest", bool(model_entry))
        items = model_entry.get("items", {})
        check("fake/model has 4 items", len(items) == 4, detail=f"got {len(items)}")

        n_with_pt = sum(1 for it in items.values() if it.get("per_token_path"))
        check("all 4 items have per_token_path", n_with_pt == 4, detail=f"got {n_with_pt}")

        n_with_att = sum(1 for it in items.values() if it.get("attention_path"))
        check("2 items have attention_path", n_with_att == 2, detail=f"got {n_with_att}")

        # Spot-check t1 (which has both arms).
        t1 = items.get("t1", {})
        pt_path = topology_dir / t1["per_token_path"] if t1.get("per_token_path") else None
        check("t1 per_token .npz exists", bool(pt_path and pt_path.exists()))
        if pt_path and pt_path.exists():
            z = np.load(pt_path)
            keys = set(z.files)
            dim_keys = {k for k in keys if k.startswith("dim_")}
            check(
                "t1 per_token has dim_0, dim_1, dim_2",
                dim_keys == {"dim_0", "dim_1", "dim_2"},
                detail=f"got {sorted(dim_keys)}",
            )
            # Diameter is the per-token normalization scale (protocol Section 4.2):
            # a positive scalar, not a diagram.
            check(
                "t1 per_token has a 'diameter' key",
                "diameter" in keys,
                detail=f"got {sorted(keys)}",
            )
            if "diameter" in keys:
                diam = z["diameter"]
                check(
                    "t1 per_token diameter is a positive scalar",
                    diam.ndim == 0 and float(diam) > 0,
                    detail=f"got shape {diam.shape}, value {float(diam) if diam.size else 'n/a'}",
                )
            for k in sorted(dim_keys):
                arr = z[k]
                check(
                    f"t1 per_token {k} is (n, 2) array",
                    arr.ndim == 2 and arr.shape[1] == 2,
                    detail=f"got shape {arr.shape}",
                )
            check(
                "t1 per_token H_0 has features (random points yield components)",
                z["dim_0"].shape[0] > 0,
                detail=f"got {z['dim_0'].shape[0]} features",
            )

        att_path = topology_dir / t1["attention_path"] if t1.get("attention_path") else None
        check("t1 attention .npz exists", bool(att_path and att_path.exists()))
        if att_path and att_path.exists():
            z = np.load(att_path)
            keys = set(z.files)
            check(
                "t1 attention has 'densities' and 'layers'",
                "densities" in keys and "layers" in keys,
            )
            tk_keys = {k for k in keys if k.startswith("thresh_")}
            # 3 thresholds x 3 dims (0, 1, 2) = 9
            check(
                "t1 attention has 9 thresh_*_dim_* keys (3 thresholds x 3 dims)",
                len(tk_keys) == 9,
                detail=f"got {len(tk_keys)}",
            )
            for k in sorted(tk_keys):
                arr = z[k]
                check(
                    f"t1 attention {k} is (n, 2) array",
                    arr.ndim == 2 and arr.shape[1] == 2,
                    detail=f"got shape {arr.shape}",
                )
            check(
                "t1 attention densities array has length 3",
                z["densities"].shape == (3,),
                detail=f"got shape {z['densities'].shape}",
            )
            check(
                "t1 attention layers array has length 4",
                z["layers"].shape == (4,),
                detail=f"got shape {z['layers'].shape}",
            )

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("All schema checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
