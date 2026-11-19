#!/usr/bin/env python
"""
Phase 1: Build the experimental corpus.

Loads NLI source datasets, filters to contradiction and matched-control
subsets, generates synthetic contradictions, and writes the corpus
components as JSONL files for the inference phase to consume.

The matched control is the primary comparison partner. Per
docs/analysis_protocol.md Section 7 it should be the NLI *neutral* item (a
coherent semantic change that is not a contradiction); set the config's
baseline.labels to ["neutral"] to honor that. The contradiction-to-control
pairing is written explicitly to pairs.jsonl so the paired analysis in
scripts/04 can reconstruct it; baseline.jsonl carries the control items
themselves for the inference and topology stages.

Cost: $0, runs locally in ~5-15 minutes depending on download speeds
for the source NLI datasets.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus.baseline_matching import MatchingConfig, find_matched_baseline
from src.corpus.item import save_corpus
from src.corpus.loaders import filter_by_agreement, filter_by_label, load_source
from src.corpus.synthetic import generate_synthetic_contradictions
from src.utils.config import ensure_dir, load_config
from src.utils.logging import get_logger, timed_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                       help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus_cfg = cfg["corpus"]
    output_dir = ensure_dir(corpus_cfg["output_dir"])
    log = get_logger("build_corpus", log_dir=output_dir / "logs")

    seed = cfg["experiment"]["random_seed"]
    log.info(f"Building corpus with seed={seed}, output_dir={output_dir}")

    # Phase 1a: Load contradiction-bearing items
    contradiction_items = []
    with timed_phase(log, "load_contradiction_sources"):
        for source in corpus_cfg["contradiction"]["sources"]:
            log.info(f"Loading source: {source}")
            items = load_source(source, split="train")
            items = filter_by_label(items, {"contradiction"})
            min_agreement = corpus_cfg["contradiction"].get("min_annotator_agreement", 0)
            if min_agreement > 0:
                items = filter_by_agreement(items, min_agreement)
            log.info(f"  {source}: {len(items)} contradictions after filtering")
            contradiction_items.extend(items)

    target_contradiction = corpus_cfg["contradiction"]["target_size"]
    if len(contradiction_items) < target_contradiction:
        log.warning(f"Only {len(contradiction_items)} contradictions available; "
                   f"target was {target_contradiction}")
    contradiction_items = contradiction_items[:target_contradiction]
    log.info(f"Final contradiction count: {len(contradiction_items)}")
    save_corpus(contradiction_items, output_dir / "contradiction.jsonl")

    # Phase 1b: Build matched control and record the pairing
    with timed_phase(log, "build_matched_control"):
        candidate_pool = []
        for source in corpus_cfg["baseline"]["sources"]:
            items = load_source(source, split="train")
            items = filter_by_label(items, set(corpus_cfg["baseline"]["labels"]))
            candidate_pool.extend(items)
        log.info(f"Candidate pool size: {len(candidate_pool)}")

        match_cfg = MatchingConfig(
            length_tolerance=corpus_cfg["baseline"]["matching"]["length_tolerance"],
            vocab_overlap_min=corpus_cfg["baseline"]["matching"]["vocab_overlap_min"],
            syntax_complexity_tolerance=corpus_cfg["baseline"]["matching"]["syntax_complexity_tolerance"],
        )
        pairs, unmatched = find_matched_baseline(
            contradiction_items, candidate_pool, match_cfg, seed=seed
        )
        log.info(f"Matched pairs: {len(pairs)}, unmatched: {len(unmatched)}")

        # Control items, for the inference and topology stages.
        control_items = [p.control for p in pairs]
        save_corpus(control_items, output_dir / "baseline.jsonl")

        # The explicit contradiction-to-control pairing, for scripts/04.
        with (output_dir / "pairs.jsonl").open("w") as f:
            for p in pairs:
                f.write(json.dumps({
                    "pair_id": p.pair_id,
                    "contradiction_id": p.contradiction.id,
                    "control_id": p.control.id,
                    "contradiction_type": p.contradiction_type,
                    "match_score": p.match_score,
                }) + "\n")
        log.info(f"Wrote pairing to {output_dir / 'pairs.jsonl'}")

        if unmatched:
            with (output_dir / "unmatched.jsonl").open("w") as f:
                for entry in unmatched:
                    f.write(json.dumps(entry) + "\n")

    # Phase 1c: Generate synthetic contradictions
    with timed_phase(log, "generate_synthetic"):
        synthetic = generate_synthetic_contradictions(
            n=corpus_cfg["synthetic"]["target_size"],
            types=corpus_cfg["synthetic"]["types"],
            seed=seed,
        )
        log.info(f"Generated {len(synthetic)} synthetic contradictions")
        save_corpus(synthetic, output_dir / "synthetic.jsonl")

    log.info(f"Corpus build complete. Files in {output_dir}")
    log.info("Next: python scripts/02_inference.py --config <config>")


if __name__ == "__main__":
    main()
