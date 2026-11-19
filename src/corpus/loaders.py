"""
Loaders for NLI benchmark datasets via Hugging Face datasets library.

Each loader produces a list of Item objects with normalized labels and
preserved annotator-agreement metadata where available.

Important:
  - SNLI uses 5-annotator labels; label_id metadata in raw data identifies
    the per-annotator votes. We compute agreement from the annotator_labels
    field where present.
  - MultiNLI matched/mismatched: we load both, marking source accordingly.
  - DocNLI is document-level NLI; we treat each premise-hypothesis pair as
    an instance even though premises can be multi-sentence.
"""

from __future__ import annotations

from collections import Counter

from .item import Item, VALID_LABELS


# Label normalization maps. Hugging Face datasets uses integer label ids;
# we normalize to string labels matching VALID_LABELS.
_SNLI_LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
    -1: None,  # SNLI uses -1 for examples with no gold label (low agreement)
}

_MULTI_NLI_LABEL_MAP = _SNLI_LABEL_MAP  # same encoding

_DOC_NLI_LABEL_MAP = {
    0: "entailment",
    1: "contradiction",   # DocNLI is binary, no "neutral" class
}


def _compute_agreement(annotator_labels: list[int]) -> int:
    """How many annotators agreed on the most common label (max over labels)."""
    if not annotator_labels:
        return 0
    counts = Counter(annotator_labels)
    return counts.most_common(1)[0][1]


def load_snli(split: str = "train", max_items: int | None = None) -> list[Item]:
    """Load Stanford NLI. Requires huggingface datasets installed."""
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/snli", split=split, streaming=False)
    items = []
    for i, row in enumerate(ds):
        if max_items is not None and i >= max_items:
            break
        label = _SNLI_LABEL_MAP.get(row["label"])
        if label is None:
            continue
        annotator_labels = row.get("annotator_labels", [])
        agreement = _compute_agreement(annotator_labels) if annotator_labels else None
        items.append(Item(
            id=f"snli_{split}_{i}",
            premise=row["premise"],
            hypothesis=row["hypothesis"],
            label=label,
            source="snli",
            annotator_agreement=agreement,
            metadata={"raw_label_id": row["label"]},
        ))
    return items


def load_multi_nli(split: str = "train", max_items: int | None = None) -> list[Item]:
    """Load Multi-Genre NLI. Use 'validation_matched' or 'validation_mismatched' for eval splits."""
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/multi_nli", split=split, streaming=False)
    items = []
    for i, row in enumerate(ds):
        if max_items is not None and i >= max_items:
            break
        label = _MULTI_NLI_LABEL_MAP.get(row["label"])
        if label is None:
            continue
        # MultiNLI validation splits have 5-annotator labels; train has 1
        annotator_labels = row.get("annotator_labels", [])
        agreement = _compute_agreement(annotator_labels) if annotator_labels else None
        items.append(Item(
            id=f"multinli_{split}_{i}",
            premise=row["premise"],
            hypothesis=row["hypothesis"],
            label=label,
            source="multi_nli",
            annotator_agreement=agreement,
            metadata={"genre": row.get("genre", ""), "raw_label_id": row["label"]},
        ))
    return items


def load_doc_nli(split: str = "train", max_items: int | None = None) -> list[Item]:
    """Load DocNLI document-level NLI."""
    from datasets import load_dataset
    ds = load_dataset("saattrupdan/doc-nli", split=split, streaming=False)
    items = []
    for i, row in enumerate(ds):
        if max_items is not None and i >= max_items:
            break
        label = _DOC_NLI_LABEL_MAP.get(row["label"])
        if label is None:
            continue
        items.append(Item(
            id=f"docnli_{split}_{i}",
            premise=row["premise"],
            hypothesis=row["hypothesis"],
            label=label,
            source="doc_nli",
            metadata={"raw_label_id": row["label"]},
        ))
    return items


def load_anli(split: str = "train", max_items: int | None = None) -> list[Item]:
    """Load Adversarial NLI (ANLI).

    ANLI premises are paragraph-length contexts (Wikipedia, HotpotQA), which is
    why it is a candidate for supporting higher-dimensional homology that
    sentence-level NLI cannot. Labels use the SNLI/MNLI encoding
    (0=entailment, 1=neutral, 2=contradiction), so the three-way contradiction
    and neutral classes are clean, unlike DocNLI's binary labels.

    ANLI has no plain 'train' split; it is collected in three rounds with
    separate train_r1/train_r2/train_r3 splits. The default 'train'
    concatenates all three rounds.
    """
    from datasets import load_dataset
    hf_split = "train_r1+train_r2+train_r3" if split == "train" else split
    ds = load_dataset("facebook/anli", split=hf_split)
    items = []
    for i, row in enumerate(ds):
        if max_items is not None and i >= max_items:
            break
        label = _SNLI_LABEL_MAP.get(row["label"])
        if label is None:
            continue
        items.append(Item(
            id=f"anli_{split}_{i}",
            premise=row["premise"],
            hypothesis=row["hypothesis"],
            label=label,
            source="anli",
            metadata={"raw_label_id": row["label"], "uid": row.get("uid", "")},
        ))
    return items


_LOADERS = {
    "snli": load_snli,
    "multi_nli": load_multi_nli,
    "doc_nli": load_doc_nli,
    "anli": load_anli,
}


def load_source(source: str, split: str = "train",
                max_items: int | None = None) -> list[Item]:
    """Dispatch to the appropriate loader by source name."""
    if source not in _LOADERS:
        raise ValueError(f"Unknown source: {source}. Available: {list(_LOADERS)}")
    return _LOADERS[source](split=split, max_items=max_items)


def filter_by_label(items: list[Item], labels: set[str]) -> list[Item]:
    """Keep only items with label in the given set."""
    return [it for it in items if it.label in labels]


def filter_by_agreement(items: list[Item], min_agreement: int) -> list[Item]:
    """Keep only items with annotator agreement >= min_agreement.
    Items without agreement metadata are kept (no filter applied)."""
    return [it for it in items
            if it.annotator_agreement is None or it.annotator_agreement >= min_agreement]
