"""
Synthetic contradiction generation.

Programmatically generates pairs of sentences that differ in a single logical
operator, numeric quantity, or negation. The synthetic component is the
controlled-construction check on the natural-language results: if the
dimensional-specificity signal appears in the synthetic component but not in
the natural-language component, that is a falsification condition (Section 4.2).

Three templates implemented:
  - single_operator_flip: swaps a logical operator (all <-> none, always <-> never)
  - single_numeric_change: swaps a quantity (3 <-> 5, 10% <-> 90%)
  - single_negation: inserts or removes a negation
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .item import Item


# Premise templates for each operator type.
# The premise is generated and the contradiction is produced by applying
# the template's transformation.

OPERATOR_PREMISES = [
    "All birds in the area can fly long distances.",
    "All employees received the year-end bonus.",
    "All members of the committee approved the proposal.",
    "All trains arrived at the station on time.",
    "All students passed the final examination.",
    "All vehicles in the lot are electric.",
    "Every door in the building was locked.",
    "Every package was delivered before noon.",
    "Every guest received a complimentary drink.",
    "Every flight departed without delay.",
]

OPERATOR_FLIPS = [
    ("All ", "None of the "),
    ("Every ", "None of the "),
    ("always", "never"),
    ("everyone", "no one"),
    ("everywhere", "nowhere"),
]


NUMERIC_PREMISES = [
    ("The committee has 7 members.", "The committee has 12 members."),
    ("The team scored 23 points in the first half.", "The team scored 47 points in the first half."),
    ("The package weighs 4 pounds.", "The package weighs 9 pounds."),
    ("The conference will host 200 attendees.", "The conference will host 800 attendees."),
    ("The building has 15 floors.", "The building has 30 floors."),
    ("The factory produces 1000 units per day.", "The factory produces 5000 units per day."),
    ("Only 25% of voters supported the measure.", "Over 75% of voters supported the measure."),
    ("The project will take 3 months to complete.", "The project will take 18 months to complete."),
    ("The contract is worth $50,000.", "The contract is worth $500,000."),
    ("The meeting starts at 9 AM.", "The meeting starts at 3 PM."),
]


NEGATION_PREMISES = [
    "The store is open on Sundays.",
    "The professor agreed with the conclusion.",
    "The flight will arrive before midnight.",
    "The medication causes drowsiness.",
    "The witness recognized the defendant.",
    "The construction will affect traffic patterns.",
    "The committee approved the new budget.",
    "The vaccine is available without a prescription.",
    "The contract includes a non-compete clause.",
    "The investigation found evidence of wrongdoing.",
]


def _generate_operator_flip(premise: str, seed_offset: int) -> tuple[str, str] | None:
    """Find an applicable flip in the premise and apply it."""
    for old, new in OPERATOR_FLIPS:
        if old in premise:
            hypothesis = premise.replace(old, new, 1)
            return premise, hypothesis
    return None


def _generate_numeric_change(premise_pair: tuple[str, str]) -> tuple[str, str]:
    """The numeric premise list is pre-paired."""
    return premise_pair


def _generate_negation(premise: str) -> tuple[str, str]:
    """Insert 'does not' / 'is not' / 'will not' to produce the contradictory hypothesis."""
    # Look for common verbs to negate
    negation_patterns = [
        (" is ", " is not "),
        (" was ", " was not "),
        (" will ", " will not "),
        (" can ", " cannot "),
        (" causes ", " does not cause "),
        (" includes ", " does not include "),
        (" agreed ", " did not agree "),
        (" found ", " did not find "),
        (" recognized ", " did not recognize "),
        (" approved ", " did not approve "),
        (" affect ", " not affect "),
    ]
    for old, new in negation_patterns:
        if old in premise:
            return premise, premise.replace(old, new, 1)
    return None


def generate_synthetic_contradictions(
    n: int,
    types: list[str],
    seed: int = 42,
) -> list[Item]:
    """
    Generate n synthetic contradiction items, balanced across requested types.

    types is a list from {"single_operator_flip", "single_numeric_change",
    "single_negation"}. n is divided as evenly as possible across types.
    """
    rng = random.Random(seed)
    items_per_type = max(1, n // len(types))
    items: list[Item] = []
    item_idx = 0

    for type_name in types:
        for _ in range(items_per_type):
            generated = None
            attempts = 0
            while generated is None and attempts < 10:
                attempts += 1
                if type_name == "single_operator_flip":
                    premise = rng.choice(OPERATOR_PREMISES)
                    generated = _generate_operator_flip(premise, item_idx)
                elif type_name == "single_numeric_change":
                    premise_pair = rng.choice(NUMERIC_PREMISES)
                    generated = _generate_numeric_change(premise_pair)
                elif type_name == "single_negation":
                    premise = rng.choice(NEGATION_PREMISES)
                    generated = _generate_negation(premise)
                else:
                    raise ValueError(f"Unknown synthetic type: {type_name}")

            if generated is None:
                continue

            premise, hypothesis = generated
            items.append(Item(
                id=f"synthetic_{type_name}_{item_idx}",
                premise=premise,
                hypothesis=hypothesis,
                label="contradiction",
                source="synthetic",
                synthetic_template=type_name,
            ))
            item_idx += 1

    # Trim or pad to exactly n
    items = items[:n]
    return items
