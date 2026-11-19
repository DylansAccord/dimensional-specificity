"""
Llama-3.1-8B loading with 4-bit quantization.

We use bitsandbytes 4-bit quantization to fit Llama-3.1-8B (16GB in FP16)
into a single A10 24GB or smaller GPU. Quantization affects attention
patterns slightly; the dry-run pipeline validates that the topology signal
survives quantization before the full experiment runs.

Important: we load with output_attentions enabled so that attention matrices
are returned in the forward pass output. This adds memory overhead compared
to standard inference, hence the smaller batch sizes in the configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LoadedModel:
    """Wrapper bundling the model, tokenizer, and key metadata."""
    model: object       # transformers.PreTrainedModel
    tokenizer: object   # transformers.PreTrainedTokenizer
    name: str
    hidden_size: int
    num_layers: int
    device: str


def load_llama(model_name: str, hf_token: str,
              quantization: str = "4bit",
              revision: Optional[str] = None,
              device_map: str = "auto",
              cache_dir: Optional[str] = None) -> LoadedModel:
    """
    Load a Llama-3 model with the specified quantization.

    quantization: "4bit", "8bit", or "fp16"
    revision: HF commit SHA to pin the weights/tokenizer (analysis_protocol
        Section 10). None loads the default branch.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs = {
        "token": hf_token,
        "revision": revision,  # SHA pin, analysis_protocol Section 10
        "device_map": device_map,
        "output_attentions": False,  # set per-call in extract_attention
        "trust_remote_code": False,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "fp16":
        kwargs["torch_dtype"] = torch.float16
    else:
        raise ValueError(f"Unknown quantization: {quantization}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token,
                                              revision=revision,
                                              cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    config = model.config
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        name=model_name,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        device=str(next(model.parameters()).device),
    )
