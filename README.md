# Dimensional specificity of contradiction in language-model representations

This repository releases the full materials for a preregistered test of one claim: that a language model processing a contradiction produces a persistent-homology signature concentrated at a single homological dimension within H2–H4 of its token-level representation. It contains the preprint, the dated preregistration written on 2026-06-30 before any model output existed, the Stage 1 corpus manifests and analysis results, and the pipeline code with pinned dependencies. The exploratory first stage (150 ANLI contradiction–neutral pairs, 86 surviving matching, Llama-3.1-8B base at revision `d04e592`, layer 24 of 32, seed 42) returned a null: no dimension approached the preregistered go/no-go criterion (largest |d_z| = 0.06, all permutation p > 0.59, Holm-adjusted p = 1.0), and the confirmatory stage was therefore not run. The null and its principal confound — a single, blindly committed extraction depth — are reported under the protocol's own falsification structure. arXiv link: TBD.

## Layout

| Path | Contents |
|---|---|
| `docs/dimensional_specificity_preprint.md` | The paper |
| `docs/analysis_protocol.md` | The preregistration (written 2026-06-30, with amendment log) |
| `docs/budget.md` | Compute-cost plan the protocol's Stage 2 sizing rule references |
| `data/corpus_stage1/` | Pair manifests (source-item IDs), unmatched log, synthetic items |
| `results/analysis_stage1/results.json` | Canonical Stage 1 results (re-run of 2026-08-19) |
| `results/topology_stage1/topology_manifest.json` | Per-item topology manifest |
| `scripts/`, `src/` | Pipeline, stages 00–04, plus schema validators |
| `configs/stage1.yaml` | Stage 1 configuration, including the model SHA pin |
| `requirements.txt` | Pinned dependencies |

## Corpus note

`data/corpus_stage1/` ships **manifests, not source text**: pair records reference ANLI items by their upstream IDs and UUIDs. ANLI is distributed by its own authors under its own terms; obtain it from the original source and rebuild with `scripts/01_build_corpus.py` (seed 42) to reproduce the exact 86-pair set. `synthetic.jsonl` is original to this work.

## Licenses

Code is released under the MIT License (`LICENSE`). Text and figures — the preprint and the preregistration — are released under CC BY 4.0 (`LICENSE-docs`).
