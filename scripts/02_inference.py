"""
Inference driver: extract per-token embeddings and layer-wise attention
from a corpus of NLI items, for one or more models, per a YAML config.

This script's outputs are the input to scripts/03_topology.py. The two are
linked by a manifest.json written to inference.output_dir.

Usage:
    python scripts/02_inference.py --config configs/dry_run.yaml

Requires GPU at run time. Local schema validation without GPU:
    python scripts/_validate_02_schema.py

Items are classified into (contradiction, baseline, synthetic) by:
  - 'synthetic' if Item.synthetic_template is set
  - else 'contradiction' if Item.label == "contradiction"
  - else 'baseline'
Per-token embeddings are extracted for every item. Attention is extracted
only for the subset configured under corpus.attention_graph_subset.

Manifest schema (manifest.json):
{
  "config_name": str,
  "config_path": str (relative to repo root),
  "corpus_dir": str,
  "output_dir": str (config value, informational),
  "n_items_total": int,
  "attention_layer_range": [int, int],
  "n_attention_subset": int,
  "models": {
    <model_name>: {
      "name": str,
      "role": str,
      "quantization": str,
      "hidden_size": int,
      "num_layers": int,
      "items": {
        <item_id>: {
          "id": str,
          "label": str,
          "source": str,
          "class": "contradiction" | "baseline" | "synthetic",
          "embedding_path": str | null,  # relative to output_dir
          "attention_path": str | null,  # relative to output_dir
          "error": str | null            # only present on failure
        }
      }
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Make src importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.corpus.item import Item, load_corpus
from src.inference.extraction import (
    extract_attention,
    extract_embeddings,
    save_attention,
    save_embedding,
)
from src.inference.model_loader import load_llama
from src.utils.config import ensure_dir, get_hf_token, load_config

logger = logging.getLogger("murmur.inference")


def classify_item(item: Item) -> str:
    """Return one of 'synthetic', 'contradiction', or 'baseline'."""
    if item.synthetic_template is not None:
        return "synthetic"
    if item.label == "contradiction":
        return "contradiction"
    return "baseline"


def load_all_items(corpus_dir: Path) -> list[Item]:
    """Load every .jsonl file in corpus_dir as a flat list of Items.

    Deduplicates by Item.id. Raises FileNotFoundError if no .jsonl files exist.
    """
    # pairs.jsonl and unmatched.jsonl are pairing manifests, not item records
    # (they carry pair_id and no premise/hypothesis); never load them as Items.
    manifest_files = {"pairs.jsonl", "unmatched.jsonl"}
    files = sorted(p for p in corpus_dir.glob("*.jsonl") if p.name not in manifest_files)
    if not files:
        raise FileNotFoundError(
            f"No .jsonl files found in {corpus_dir}. "
            "Run scripts/01_build_corpus.py first."
        )
    items: list[Item] = []
    seen_ids: set[str] = set()
    for path in files:
        for item in load_corpus(path):
            if item.id in seen_ids:
                logger.warning("Duplicate item.id %s; keeping first occurrence", item.id)
                continue
            seen_ids.add(item.id)
            items.append(item)
    logger.info("Loaded %d items from %d corpus file(s)", len(items), len(files))
    return items


def pick_attention_subset(
    items: list[Item],
    subset_cfg: dict,
    seed: int,
) -> set[str]:
    """Sample item IDs that should receive attention extraction.

    subset_cfg maps class name ('contradiction', 'baseline', 'synthetic') to
    a target count. Sampling is deterministic given seed.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[Item]] = {
        "contradiction": [],
        "baseline": [],
        "synthetic": [],
    }
    for item in items:
        by_class[classify_item(item)].append(item)

    chosen: set[str] = set()
    for cls, target in subset_cfg.items():
        pool = by_class.get(cls, [])
        if not pool:
            logger.warning("No items in class '%s'; skipping attention subset for it", cls)
            continue
        if target >= len(pool):
            chosen.update(it.id for it in pool)
            logger.info("Attention subset: all %d items from '%s'", len(pool), cls)
        else:
            sampled = rng.sample(pool, target)
            chosen.update(it.id for it in sampled)
            logger.info(
                "Attention subset: sampled %d of %d items from '%s'",
                target, len(pool), cls,
            )
    return chosen


def model_safename(model_name: str) -> str:
    """Convert an HF model name to a filesystem-safe directory name."""
    return model_name.replace("/", "__")


def run_model(
    model_cfg: dict,
    items: list[Item],
    attention_ids: set[str],
    extract_per_token: bool,
    extract_attn: bool,
    attention_layer_range: list[int],
    per_token_layer: int,
    output_dir: Path,
    hf_token: str,
    hf_cache_dir: str | None,
) -> dict:
    """Load one model, run extractions, return its manifest entry.

    The model is freed before this function returns so the next model can be
    loaded without OOM.
    """
    model_name = model_cfg["name"]
    role = model_cfg.get("role", "unspecified")
    quantization = model_cfg.get("quantization", "4bit")

    model_dir = ensure_dir(output_dir / model_safename(model_name))

    logger.info("Loading %s (role=%s, quantization=%s)", model_name, role, quantization)
    loaded = load_llama(
        model_name=model_name,
        hf_token=hf_token,
        quantization=quantization,
        revision=model_cfg.get("revision"),
        cache_dir=hf_cache_dir,
    )
    logger.info(
        "Loaded: hidden_size=%d num_layers=%d device=%s",
        loaded.hidden_size, loaded.num_layers, loaded.device,
    )

    item_entries: dict[str, dict] = {}
    n_emb = 0
    n_att = 0

    for i, item in enumerate(items):
        entry: dict = {
            "id": item.id,
            "label": item.label,
            "source": item.source,
            "class": classify_item(item),
            "embedding_path": None,
            "attention_path": None,
        }
        try:
            if extract_per_token:
                emb_path = model_dir / f"{item.id}__embeddings.npz"
                emb_result = extract_embeddings(item, loaded, layer=per_token_layer)
                save_embedding(emb_result, str(emb_path))
                entry["embedding_path"] = str(emb_path.relative_to(output_dir))
                n_emb += 1

            if extract_attn and item.id in attention_ids:
                att_path = model_dir / f"{item.id}__attention.npz"
                att_result = extract_attention(
                    item, loaded, layer_range=tuple(attention_layer_range),
                )
                save_attention(att_result, str(att_path))
                entry["attention_path"] = str(att_path.relative_to(output_dir))
                n_att += 1
        except Exception as e:
            logger.exception("Failed on item %s: %s", item.id, e)
            entry["error"] = str(e)

        item_entries[item.id] = entry

        if (i + 1) % 25 == 0 or (i + 1) == len(items):
            logger.info("  progress: %d / %d items", i + 1, len(items))

    logger.info(
        "Done with %s: %d embeddings, %d attention matrices written",
        model_name, n_emb, n_att,
    )

    # Free model memory before next model.
    del loaded
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # Re-load just for the metadata fields (in production we cached them above
    # before del; here we just record what we have).
    return {
        "name": model_name,
        "role": role,
        "quantization": quantization,
        "items": item_entries,
    }


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    seed = cfg["experiment"]["random_seed"]

    corpus_dir = Path(cfg["corpus"]["output_dir"])
    items = load_all_items(corpus_dir)

    inf_cfg = cfg["inference"]
    output_dir = ensure_dir(Path(inf_cfg["output_dir"]))

    reps_cfg = inf_cfg["representations"]
    extract_per_token = bool(reps_cfg.get("extract_per_token_embeddings", True))
    extract_attn = bool(reps_cfg.get("extract_attention", True))
    attention_layer_range = list(reps_cfg.get("attention_layer_range", [22, 32]))
    per_token_layer = int(reps_cfg.get("per_token_layer", 24))

    if reps_cfg.get("extract_sentence_embeddings"):
        logger.warning(
            "extract_sentence_embeddings=True but no sentence-embedding function "
            "exists in src/inference/extraction.py. Skipping. Per-token + attention "
            "are the two designed arms; sentence embeddings would need adding to "
            "extraction.py first if you actually want them."
        )

    attention_subset_cfg = cfg["corpus"].get("attention_graph_subset", {}) or {}
    if extract_attn and attention_subset_cfg:
        attention_ids = pick_attention_subset(items, attention_subset_cfg, seed)
        logger.info("Attention subset: %d items selected", len(attention_ids))
    else:
        attention_ids = set()
        if extract_attn:
            logger.warning(
                "extract_attention=True but no corpus.attention_graph_subset "
                "configured; no attention will be extracted"
            )

    hf_token = get_hf_token(cfg)
    hf_cache_dir = cfg["huggingface"].get("cache_dir")

    manifest: dict = {
        "config_name": cfg["experiment"]["name"],
        "config_path": str(Path(config_path)),
        "corpus_dir": str(corpus_dir),
        "output_dir": str(inf_cfg["output_dir"]),
        "n_items_total": len(items),
        "attention_layer_range": attention_layer_range,
        "per_token_layer": per_token_layer,
        "n_attention_subset": len(attention_ids),
        "models": {},
    }

    for model_cfg in inf_cfg["models"]:
        entry = run_model(
            model_cfg=model_cfg,
            items=items,
            attention_ids=attention_ids,
            extract_per_token=extract_per_token,
            extract_attn=extract_attn,
            attention_layer_range=attention_layer_range,
            per_token_layer=per_token_layer,
            output_dir=output_dir,
            hf_token=hf_token,
            hf_cache_dir=hf_cache_dir,
        )
        manifest["models"][model_cfg["name"]] = entry

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest: %s", manifest_path)
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
        description="Run inference: per-token embeddings + attention extraction.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    sys.exit(main(args.config))
