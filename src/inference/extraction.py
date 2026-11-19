"""
Representation extraction from Llama-3.1-8B.

Two extraction modes:
  - extract_embeddings: per-token contextual embeddings as a point cloud
    (for the sentence-embedding arm — Vietoris-Rips persistence input)
  - extract_attention: layer-wise attention matrices as weighted graphs
    (for the attention-graph arm — zigzag persistence input)

Both operate on Item.joint_text (premise and hypothesis concatenated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..corpus.item import Item
from .model_loader import LoadedModel


@dataclass
class EmbeddingResult:
    """Per-token embeddings for a single item.

    embeddings has shape (n_tokens, hidden_size). The point cloud for
    persistent homology is this array directly.
    """
    item_id: str
    embeddings: np.ndarray         # (n_tokens, hidden_size)
    token_ids: list[int]
    token_strings: list[str]


@dataclass
class AttentionResult:
    """Layer-wise attention for a single item.

    attention_matrices[i] is the attention matrix at layer i, shape
    (n_heads, n_tokens, n_tokens). Layer indices are absolute (i.e.,
    layer 22 of 32 if the config requested final-third layers).
    """
    item_id: str
    attention_matrices: dict[int, np.ndarray]   # layer_idx -> attention
    token_ids: list[int]


@torch.no_grad()
def extract_embeddings(item: Item, loaded: LoadedModel,
                       layer: int = 24,
                       max_length: int = 512) -> EmbeddingResult:
    """
    Extract per-token contextual embeddings from a chosen hidden layer.

    `layer` indexes the hidden_states tuple, which has num_layers + 1 entries:
    index 0 is the input embedding, index k is the residual stream after block k.
    The confirmatory per-token arm is locked to layer 24 of 32 (analysis_protocol
    Section 10). The previous hardcoded final-layer behavior corresponds to
    layer = -1.

    Uses Item.joint_text as input. Returns the embeddings as a numpy array
    suitable for direct use as a point cloud in Vietoris-Rips persistence.
    """
    tokenizer = loaded.tokenizer
    model = loaded.model

    inputs = tokenizer(
        item.joint_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    inputs = {k: v.to(loaded.device) for k, v in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True, output_attentions=False)
    # hidden_states is a tuple of (num_layers + 1) tensors of shape (1, n_tokens, hidden_size).
    # Index 0 is the input embedding; index k is the residual stream after block k.
    n_states = len(outputs.hidden_states)
    if not (-n_states <= layer < n_states):
        raise ValueError(f"per-token layer {layer} out of range for {n_states} hidden states")
    final_hidden = outputs.hidden_states[layer].squeeze(0).cpu().to(torch.float32).numpy()

    token_ids = inputs["input_ids"].squeeze(0).cpu().tolist()
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)

    return EmbeddingResult(
        item_id=item.id,
        embeddings=final_hidden,
        token_ids=token_ids,
        token_strings=token_strings,
    )


@torch.no_grad()
def extract_attention(item: Item, loaded: LoadedModel,
                     layer_range: tuple[int, int],
                     max_length: int = 512) -> AttentionResult:
    """
    Extract attention matrices from a range of layers.

    layer_range is (start, end) inclusive of start, exclusive of end, matching
    Python slice semantics. For the final third of a 32-layer model, pass
    (22, 32).

    Returns AttentionResult with attention_matrices keyed by absolute layer
    index. Each matrix has shape (n_heads, n_tokens, n_tokens).
    """
    tokenizer = loaded.tokenizer
    model = loaded.model
    start_layer, end_layer = layer_range
    if end_layer > loaded.num_layers:
        raise ValueError(f"end_layer {end_layer} exceeds num_layers {loaded.num_layers}")

    inputs = tokenizer(
        item.joint_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    inputs = {k: v.to(loaded.device) for k, v in inputs.items()}

    outputs = model(**inputs, output_attentions=True, output_hidden_states=False)
    # attentions is a tuple of num_layers tensors, each (1, n_heads, n_tokens, n_tokens)
    attention_matrices = {}
    for layer_idx in range(start_layer, end_layer):
        att = outputs.attentions[layer_idx].squeeze(0).cpu().to(torch.float32).numpy()
        attention_matrices[layer_idx] = att

    token_ids = inputs["input_ids"].squeeze(0).cpu().tolist()

    return AttentionResult(
        item_id=item.id,
        attention_matrices=attention_matrices,
        token_ids=token_ids,
    )


def save_embedding(result: EmbeddingResult, path: str) -> None:
    """Save embeddings as .npz with metadata."""
    np.savez_compressed(
        path,
        embeddings=result.embeddings,
        token_ids=np.array(result.token_ids),
        token_strings=np.array(result.token_strings),
        item_id=result.item_id,
    )


def save_attention(result: AttentionResult, path: str) -> None:
    """Save attention matrices as .npz."""
    save_dict = {f"layer_{k}": v for k, v in result.attention_matrices.items()}
    save_dict["token_ids"] = np.array(result.token_ids)
    save_dict["item_id"] = result.item_id
    np.savez_compressed(path, **save_dict)
