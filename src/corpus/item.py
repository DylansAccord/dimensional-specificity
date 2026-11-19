"""
Canonical corpus item type used throughout the pipeline.

Every NLI instance, whether loaded from SNLI/MultiNLI/DocNLI, generated
synthetically, or constructed as a matched baseline, is represented as an
Item with the same schema. This makes the inference and topology code
agnostic to corpus source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Item:
    """A single NLI instance.

    Fields:
      id: unique identifier (typically "source_originalid", e.g. "snli_12345")
      premise: premise sentence
      hypothesis: hypothesis sentence
      label: "contradiction", "entailment", or "neutral"
      source: which dataset/method produced this item
      contradiction_type: one of Wu 2022's six types, or None if not classified
      annotator_agreement: number of annotators that agreed on label, if available
      synthetic_template: template name if synthetic, else None
      metadata: source-specific extra fields
    """
    id: str
    premise: str
    hypothesis: str
    label: str
    source: str
    contradiction_type: Optional[str] = None
    annotator_agreement: Optional[int] = None
    synthetic_template: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def joint_text(self) -> str:
        """Premise and hypothesis concatenated for joint-embedding analysis."""
        return f"{self.premise} {self.hypothesis}"


def save_corpus(items: list[Item], path: str | Path) -> None:
    """Write items as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")


def load_corpus(path: str | Path) -> list[Item]:
    """Read items from JSONL."""
    items = []
    with Path(path).open("r") as f:
        for line in f:
            d = json.loads(line)
            items.append(Item(**d))
    return items


# Valid label values
VALID_LABELS = ("contradiction", "entailment", "neutral")

# Wu 2022 six-type taxonomy
CONTRADICTION_TYPES = (
    "antonymy",   # opposite-meaning lexical items
    "negation",   # explicit negation (not, never, no)
    "numeric",    # numerical contradiction
    "factive",    # factive vs counterfactive verbs
    "structural", # structural/syntactic contradiction
    "lexical",    # general lexical contradiction
)
