"""
Local schema validator for scripts/02_inference.py.

scripts/02 requires a GPU for real inference. The orchestration logic and
manifest schema, however, are independent of the model. This validator
mocks load_llama, extract_embeddings, and extract_attention with CPU-only
fakes, runs scripts/02's main() against a small synthetic corpus in a temp
directory, and verifies that the manifest and output files have the shape
scripts/03_topology.py will consume.

No GPU required. Exits 0 on PASS, 1 on FAIL. Cleans up its tempdir on exit.

Usage:
    python scripts/_validate_02_schema.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.corpus.item import Item, save_corpus

# Import the submodules we'll patch so that unittest.mock can resolve their
# dotted paths via attribute access. (mock.patch uses pkgutil.resolve_name,
# which does NOT trigger import; it relies on the submodule already being
# attached as an attribute of its parent package.)
import src.inference.model_loader  # noqa: F401
import src.inference.extraction  # noqa: F401

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ---- Fakes ----


class _FakeLoadedModel:
    """Stand-in for LoadedModel. Attributes only; no real model."""

    def __init__(self, name: str):
        self.name = name
        self.model = object()
        self.tokenizer = object()
        self.hidden_size = 4096
        self.num_layers = 32
        self.device = "cpu"


def _fake_load_llama(model_name, hf_token, quantization="4bit",
                     revision=None, device_map="auto", cache_dir=None):
    return _FakeLoadedModel(model_name)


def _fake_extract_embeddings(item, loaded, layer=24, max_length=512):
    from src.inference.extraction import EmbeddingResult
    n_tokens = 8
    return EmbeddingResult(
        item_id=item.id,
        embeddings=np.random.randn(n_tokens, loaded.hidden_size).astype(np.float32),
        token_ids=list(range(n_tokens)),
        token_strings=[f"tok{i}" for i in range(n_tokens)],
    )


def _fake_extract_attention(item, loaded, layer_range, max_length=512):
    from src.inference.extraction import AttentionResult
    n_tokens = 8
    n_heads = 4
    start, end = layer_range
    attention_matrices = {
        i: np.random.rand(n_heads, n_tokens, n_tokens).astype(np.float32)
        for i in range(start, end)
    }
    return AttentionResult(
        item_id=item.id,
        attention_matrices=attention_matrices,
        token_ids=list(range(n_tokens)),
    )


# ---- Fixtures ----


def _make_fake_corpus(corpus_dir: Path) -> list[Item]:
    items = [
        Item(
            id="t1", premise="The cat is black.", hypothesis="The cat is white.",
            label="contradiction", source="snli", annotator_agreement=5,
        ),
        Item(
            id="t2", premise="A man walks.", hypothesis="A person moves.",
            label="entailment", source="snli", annotator_agreement=5,
        ),
        Item(
            id="t3", premise="It is raining.", hypothesis="Someone is happy.",
            label="neutral", source="snli", annotator_agreement=4,
        ),
        Item(
            id="t4", premise="The light is on.", hypothesis="The light is not on.",
            label="contradiction", source="synth_negation",
            synthetic_template="single_negation",
        ),
        Item(
            id="t5", premise="A dog barks.", hypothesis="A dog does not bark.",
            label="contradiction", source="synth_negation",
            synthetic_template="single_negation",
        ),
    ]
    save_corpus(items, corpus_dir / "test.jsonl")
    return items


def _make_fake_config(tmpdir: Path) -> Path:
    cfg = {
        "experiment": {"name": "schema_test", "random_seed": 42},
        "huggingface": {"token": "hf_fake_for_test"},
        "corpus": {
            "output_dir": str(tmpdir / "corpus"),
            "attention_graph_subset": {
                "contradiction": 1, "baseline": 1, "synthetic": 1,
            },
        },
        "inference": {
            "output_dir": str(tmpdir / "embeddings"),
            "models": [
                {"name": "fake/model-A", "role": "base", "quantization": "4bit"},
                {"name": "fake/model-B", "role": "instruct", "quantization": "4bit"},
            ],
            "representations": {
                "extract_per_token_embeddings": True,
                "extract_attention": True,
                "attention_layer_range": [0, 4],
                "extract_sentence_embeddings": True,  # expect: skipped with warning
            },
        },
        "topology": {},
        "analysis": {},
    }
    path = tmpdir / "schema_test.yaml"
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _load_script02():
    """Import scripts/02_inference.py as a module despite the leading digit."""
    spec = importlib.util.spec_from_file_location(
        "_script02", REPO_ROOT / "scripts" / "02_inference.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- Runner ----


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
        corpus_dir = tmpdir / "corpus"
        corpus_dir.mkdir()
        _make_fake_corpus(corpus_dir)
        cfg_path = _make_fake_config(tmpdir)

        # Patch upstream modules BEFORE importing scripts/02, so that
        # script02's `from src.inference.X import Y` resolves to the fakes.
        with patch("src.inference.model_loader.load_llama",
                   side_effect=_fake_load_llama), \
             patch("src.inference.extraction.extract_embeddings",
                   side_effect=_fake_extract_embeddings), \
             patch("src.inference.extraction.extract_attention",
                   side_effect=_fake_extract_attention):
            script02 = _load_script02()
            rc = script02.main(str(cfg_path))
            check("main() returns 0", rc == 0, detail=f"got {rc}")

        # Validate outputs.
        output_dir = tmpdir / "embeddings"
        manifest_path = output_dir / "manifest.json"
        check("manifest.json exists", manifest_path.exists())
        if not manifest_path.exists():
            print("\nFAILED.")
            return 1

        with manifest_path.open() as f:
            manifest = json.load(f)

        check("manifest has 'models' key", "models" in manifest)
        check("manifest has 2 models", len(manifest.get("models", {})) == 2)
        check("manifest.n_items_total == 5", manifest.get("n_items_total") == 5)
        check(
            "manifest.attention_layer_range == [0, 4]",
            manifest.get("attention_layer_range") == [0, 4],
        )
        check(
            "manifest.n_attention_subset > 0",
            manifest.get("n_attention_subset", 0) > 0,
        )

        for model_name in ["fake/model-A", "fake/model-B"]:
            entry = manifest.get("models", {}).get(model_name, {})
            check(f"{model_name} present in manifest", bool(entry))
            check(
                f"{model_name} has 5 item entries",
                len(entry.get("items", {})) == 5,
                detail=f"got {len(entry.get('items', {}))}",
            )

            items = entry.get("items", {})
            n_with_emb = sum(1 for it in items.values() if it.get("embedding_path"))
            n_with_att = sum(1 for it in items.values() if it.get("attention_path"))
            check(
                f"{model_name}: all 5 items have embedding_path",
                n_with_emb == 5,
                detail=f"got {n_with_emb}",
            )
            check(
                f"{model_name}: attention count matches subset size",
                n_with_att == manifest["n_attention_subset"],
                detail=f"got {n_with_att}, subset size {manifest['n_attention_subset']}",
            )

            # Spot-check: at least one embedding file actually exists on disk
            # and at least one attention file actually exists on disk.
            emb_paths_on_disk = []
            att_paths_on_disk = []
            for item_id, item_entry in items.items():
                if item_entry.get("embedding_path"):
                    full = output_dir / item_entry["embedding_path"]
                    if full.exists():
                        emb_paths_on_disk.append(full)
                if item_entry.get("attention_path"):
                    full = output_dir / item_entry["attention_path"]
                    if full.exists():
                        att_paths_on_disk.append(full)
            check(
                f"{model_name}: all embedding files written",
                len(emb_paths_on_disk) == n_with_emb,
            )
            check(
                f"{model_name}: all attention files written",
                len(att_paths_on_disk) == n_with_att,
            )

            # Open one embedding file and confirm shape.
            if emb_paths_on_disk:
                z = np.load(emb_paths_on_disk[0], allow_pickle=True)
                check(
                    f"{model_name}: embedding npz has 'embeddings' key",
                    "embeddings" in z.files,
                )
                if "embeddings" in z.files:
                    check(
                        f"{model_name}: embeddings array is 2D (n_tokens, hidden)",
                        z["embeddings"].ndim == 2,
                    )

            # Open one attention file and confirm structure.
            if att_paths_on_disk:
                z = np.load(att_paths_on_disk[0], allow_pickle=True)
                layer_keys = [k for k in z.files if k.startswith("layer_")]
                check(
                    f"{model_name}: attention npz has 4 layer_* keys",
                    len(layer_keys) == 4,
                    detail=f"got {layer_keys}",
                )

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("All schema checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
