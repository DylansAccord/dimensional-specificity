"""Llama-3.1-8B inference: sentence embeddings and attention extraction.

Two extraction modes for the two experimental arms:
  - Sentence-embedding arm: per-token contextual embeddings as point clouds
  - Attention-graph arm: layer-wise attention matrices as weighted graphs

Both use 4-bit quantization via bitsandbytes to fit in 24GB VRAM. The
quantization affects attention patterns slightly; the dry-run validates that
the topology signal survives quantization before committing the full budget.
"""
