"""
Matched-baseline construction for the dimensional-specificity experiment.

The matched baseline is the most important methodological piece of the
experiment. Superficial features of input text (sentence length, vocabulary
distribution, syntactic complexity) themselves alter the topology of
embedding point clouds. If we compare contradiction-bearing inputs against
arbitrary non-contradiction inputs, any topological difference could reflect
length or vocabulary differences rather than the contradiction signal we
want to isolate.

The matching procedure: for each contradiction-bearing item, find a
non-contradictory control item from the candidate pool whose surface features
fall within tolerance bands defined in the config. Per docs/analysis_protocol.md
Section 7 the primary control is the NLI *neutral* item (a coherent semantic
change that is not a contradiction); the candidate pool handed in by
01_build_corpus.py determines which label is used.

Critical change over the earlier version: the contradiction-to-control link
is now preserved. find_matched_baseline returns MatchedPair records, each
carrying a stable pair_id and both members, so the paired analysis downstream
(bootstrap, permutation) has an explicit, order-independent pairing. The old
version returned a flat shuffled list of controls with the link discarded,
which silently broke the matched-pair design.

Tolerance bands (from configs/experiment.example.yaml):
  length_tolerance: +/-15% of token count
  vocab_overlap_min: Jaccard >= 0.3 on content-word vocabulary
  syntax_complexity_tolerance: +/-20% of dependency-depth proxy
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .item import Item


# Simple English stop-word list for content-word extraction.
# Not perfect, but adequate for the matching heuristic.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "not", "no",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization. Strips punctuation."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def _content_words(text: str) -> set[str]:
    """Tokens that are not stopwords."""
    return {w for w in _tokenize(text) if w not in _STOPWORDS}


def _sentence_length(item: Item) -> int:
    """Token count of joint premise+hypothesis."""
    return len(_tokenize(item.joint_text))


def _vocab_jaccard(a: Item, b: Item) -> float:
    """Jaccard similarity of content-word vocabularies."""
    va = _content_words(a.joint_text)
    vb = _content_words(b.joint_text)
    if not va and not vb:
        return 1.0
    union = va | vb
    if not union:
        return 0.0
    return len(va & vb) / len(union)


def _syntactic_complexity(item: Item) -> float:
    """
    Cheap proxy for syntactic complexity: ratio of clause-introducing words
    to total tokens. Words like "because", "although", "while" indicate
    subordinate clauses. Not a true parse, but discriminates simple from
    complex constructions adequately for matching.
    """
    clause_markers = frozenset({
        "because", "although", "though", "while", "whereas", "unless",
        "until", "since", "when", "whenever", "where", "wherever", "if",
        "whether", "after", "before", "as", "that", "which", "who",
    })
    tokens = _tokenize(item.joint_text)
    if not tokens:
        return 0.0
    n_markers = sum(1 for t in tokens if t in clause_markers)
    return n_markers / len(tokens)


@dataclass
class MatchingConfig:
    length_tolerance: float = 0.15
    vocab_overlap_min: float = 0.3
    syntax_complexity_tolerance: float = 0.2


@dataclass
class MatchedPair:
    """One matched (contradiction, control) pair.

    pair_id is stable and derived from the contradiction id, so the same
    corpus produces the same pair ids across runs. contradiction_type is
    carried up from the contradiction item for type-stratified analysis.
    """
    pair_id: str
    contradiction: Item
    control: Item
    contradiction_type: Optional[str]
    match_score: float


def find_matched_baseline(
    contradiction_items: list[Item],
    candidate_pool: list[Item],
    config: MatchingConfig,
    seed: int = 42,
) -> tuple[list[MatchedPair], list[dict]]:
    """
    For each contradiction item, find a control item from the pool that
    matches on surface features. Each control is used at most once.

    Returns (pairs, unmatched_log):
      - pairs: list of MatchedPair, one per successfully matched contradiction.
        The contradiction-to-control link is explicit and order-independent.
      - unmatched_log: list of dicts for contradiction items with no suitable
        control within tolerance.

    Every contradiction item appears in exactly one of the two outputs, so
    len(pairs) + len(unmatched_log) == len(contradiction_items).

    Algorithm: for each contradiction item, score candidates by a combined
    matching score (lower = better), pick the best available within the hard
    tolerance bands. Greedy and order-dependent; the contradiction list is
    shuffled to avoid systematic bias. The shuffle no longer affects the
    pairing, which is recorded explicitly per pair.
    """
    rng = np.random.default_rng(seed)
    contradiction_indices = list(range(len(contradiction_items)))
    rng.shuffle(contradiction_indices)

    # Precompute features for candidate pool.
    candidate_features = []
    for cand in candidate_pool:
        candidate_features.append({
            "item": cand,
            "length": _sentence_length(cand),
            "content_words": _content_words(cand.joint_text),
            "syntax": _syntactic_complexity(cand),
        })

    used_candidates: set[int] = set()
    pairs: list[MatchedPair] = []
    unmatched_log: list[dict] = []

    for con_idx in contradiction_indices:
        con = contradiction_items[con_idx]
        con_length = _sentence_length(con)
        con_content = _content_words(con.joint_text)
        con_syntax = _syntactic_complexity(con)

        length_min = con_length * (1 - config.length_tolerance)
        length_max = con_length * (1 + config.length_tolerance)
        syntax_min = con_syntax - config.syntax_complexity_tolerance
        syntax_max = con_syntax + config.syntax_complexity_tolerance

        best_score = float("inf")
        best_idx = None

        for cand_idx, feats in enumerate(candidate_features):
            if cand_idx in used_candidates:
                continue
            # Hard filters.
            if not (length_min <= feats["length"] <= length_max):
                continue
            if not (syntax_min <= feats["syntax"] <= syntax_max):
                continue
            # Vocab overlap (Jaccard).
            vocab_union = con_content | feats["content_words"]
            if not vocab_union:
                jaccard = 0.0
            else:
                jaccard = len(con_content & feats["content_words"]) / len(vocab_union)
            if jaccard < config.vocab_overlap_min:
                continue
            # Combined score: weighted distance on the three axes.
            length_diff = abs(feats["length"] - con_length) / max(con_length, 1)
            syntax_diff = abs(feats["syntax"] - con_syntax)
            vocab_diff = 1 - jaccard
            score = length_diff + syntax_diff + vocab_diff
            if score < best_score:
                best_score = score
                best_idx = cand_idx

        if best_idx is None:
            unmatched_log.append({
                "contradiction_id": con.id,
                "length": con_length,
                "syntax": con_syntax,
                "reason": "no candidate within tolerance bands",
            })
            continue

        used_candidates.add(best_idx)
        control = candidate_features[best_idx]["item"]
        pairs.append(MatchedPair(
            pair_id=f"pair_{con.id}",
            contradiction=con,
            control=control,
            contradiction_type=con.contradiction_type,
            match_score=float(best_score),
        ))

    return pairs, unmatched_log


# ----- Sanity check (run: python -m src.corpus.baseline_matching) -----


def _sanity_check() -> None:
    import sys

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        marker = "PASS" if cond else "FAIL"
        tail = f": {detail}" if (detail and not cond) else ""
        print(f"  [{marker}] {name}{tail}")
        if not cond:
            failures.append(name)

    def mk(id_, premise, hypothesis, label, ctype=None):
        return Item(id=id_, premise=premise, hypothesis=hypothesis,
                    label=label, source="test", contradiction_type=ctype)

    # --- helpers unchanged ---

    c = mk("c", "the quick brown fox jumps", "the fox does not jump", "contradiction")
    check("length counts joint tokens (10)", _sentence_length(c) == 10, detail=f"got {_sentence_length(c)}")

    a = mk("a", "the brown fox runs", "the fox runs fast", "neutral")
    b = mk("b", "a completely different sentence entirely", "with nothing shared", "neutral")
    jab = _vocab_jaccard(c, a)
    check("vocab jaccard in [0,1]", 0.0 <= jab <= 1.0, detail=f"got {jab}")
    check("shared-vocab pair scores higher than disjoint pair",
          _vocab_jaccard(c, a) > _vocab_jaccard(c, b))

    # --- pairing preserved ---

    # Two contradictions, each with one good neutral match and decoys.
    con1 = mk("con1", "the brown dog chased the small cat", "the dog did not chase", "contradiction", "negation")
    con2 = mk("con2", "she bought three red apples today", "she bought five red apples today", "contradiction", "numeric")
    # Good matches: similar length, overlapping content words, low syntax markers.
    good1 = mk("good1", "the brown dog chased the small bird", "the dog chased happily", "neutral")
    good2 = mk("good2", "she bought three red apples nearby", "she sold red apples today", "neutral")
    # Decoy: wildly different length and vocab.
    decoy = mk("decoy", "x", "y", "neutral")

    cfg = MatchingConfig()
    pairs, unmatched = find_matched_baseline([con1, con2], [decoy, good1, good2], cfg, seed=1)

    check("accounting: pairs + unmatched == n contradictions",
          len(pairs) + len(unmatched) == 2, detail=f"{len(pairs)}+{len(unmatched)}")

    by_con = {p.contradiction.id: p for p in pairs}
    check("con1 was matched", "con1" in by_con)
    check("con2 was matched", "con2" in by_con)
    if "con1" in by_con:
        check("con1 paired with its good control",
              by_con["con1"].control.id == "good1",
              detail=f"got {by_con['con1'].control.id}")
    if "con2" in by_con:
        check("con2 paired with its good control",
              by_con["con2"].control.id == "good2",
              detail=f"got {by_con['con2'].control.id}")

    # pair_id derived from contradiction id, contradiction_type carried up.
    if "con1" in by_con:
        check("pair_id derived from contradiction id",
              by_con["con1"].pair_id == "pair_con1", detail=by_con["con1"].pair_id)
        check("contradiction_type carried onto pair",
              by_con["con1"].contradiction_type == "negation")

    # Each control used at most once.
    control_ids = [p.control.id for p in pairs]
    check("each control used at most once", len(control_ids) == len(set(control_ids)),
          detail=f"controls={control_ids}")

    # Matched control satisfies the hard filters relative to its contradiction.
    for p in pairs:
        cl = _sentence_length(p.contradiction)
        ctrl_l = _sentence_length(p.control)
        within_len = cl * (1 - cfg.length_tolerance) <= ctrl_l <= cl * (1 + cfg.length_tolerance)
        jac = _vocab_jaccard(p.contradiction, p.control)
        check(f"{p.pair_id}: control within length tolerance", within_len,
              detail=f"con={cl} ctrl={ctrl_l}")
        check(f"{p.pair_id}: control meets vocab overlap", jac >= cfg.vocab_overlap_min,
              detail=f"jaccard={jac:.3f}")

    # --- unmatched path ---

    lonely = mk("lonely", "supercalifragilistic expialidocious nonsense tokens here", "more unmatched gibberish", "contradiction")
    pairs2, unmatched2 = find_matched_baseline([lonely], [decoy], cfg, seed=1)
    check("contradiction with no valid control goes to unmatched, not pairs",
          len(pairs2) == 0 and len(unmatched2) == 1, detail=f"{len(pairs2)},{len(unmatched2)}")
    check("unmatched entry records the contradiction id",
          unmatched2[0]["contradiction_id"] == "lonely")

    # --- determinism ---

    pa, _ = find_matched_baseline([con1, con2], [decoy, good1, good2], cfg, seed=7)
    pb, _ = find_matched_baseline([con1, con2], [decoy, good1, good2], cfg, seed=7)
    check("same seed gives same pairing",
          {p.contradiction.id: p.control.id for p in pa}
          == {p.contradiction.id: p.control.id for p in pb})

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {failures}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    print("baseline_matching sanity check\n")
    _sanity_check()
