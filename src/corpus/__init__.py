"""NLI corpus loading, filtering, baseline matching, and synthetic generation.

The experiment requires three corpus components:
  - Contradiction-bearing (from SNLI, MultiNLI, DocNLI, filtered for annotator agreement)
  - Matched non-contradictory baseline (same sources, controlled on length/vocab/syntax)
  - Synthetic-construction (programmatically generated for controlled-construction check)

The matching step is critical: superficial features (length, vocabulary, syntax)
themselves alter the topology of embedding point clouds. Matching the baseline
on these features isolates the contradiction signal from confounds.
"""
