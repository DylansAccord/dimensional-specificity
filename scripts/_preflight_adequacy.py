#!/usr/bin/env python
"""
PH-blinded point-count preflight (docs/analysis_protocol.md Section 8).

Decides which homology dimensions are statistically testable BEFORE any
topology is computed and before any cloud spend, using token counts alone.
Because the decision is made from token counts with no persistent homology and
no separation result in view, it cannot be a post-hoc redefinition of the
hypothesis: H3/H4 are declared adequate or inconclusive here, in a dated
report committed to the repo, not after the data comes back.

The point count for an item is the number of token embeddings in its cloud,
which is exactly what the Llama tokenizer produces (special tokens excluded,
matching the per-token arm's representation). A whitespace proxy undercounts
BPE subwords, so the real tokenizer is preferred; the proxy is only an
automatic fallback when the tokenizer cannot be loaded.

Adequacy rule:
  - per item: a cloud needs at least THRESHOLD_POINTS points.
  - per cell: H3 and H4 are adequate only if at least COVERAGE_REQUIRED of the
    cell's items meet the per-item threshold. H0-H2 are always adequate (low
    dimensions need few points).

Output: <corpus.output_dir>/preflight_report.json. Cost: $0, CPU only.

Run: python scripts/_preflight_adequacy.py --config <config>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus.item import load_corpus

THRESHOLD_POINTS = 25
COVERAGE_REQUIRED = 0.80


def proxy_token_count(text: str) -> int:
    """Whitespace/word fallback count. Undercounts BPE subwords; fallback only."""
    return len(re.findall(r"\b\w+\b", text))


def load_tokenizer(model_name: str, token: str | None):
    """Return (count_fn, label). Real Llama tokenizer if it loads, else proxy.

    count_fn(text) -> int counts content tokens (special tokens excluded), which
    is what the per-token point cloud contains.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, token=token)

        def count_fn(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))

        return count_fn, model_name
    except Exception as e:  # noqa: BLE001 - any failure falls back to the proxy
        print(f"  WARNING: could not load tokenizer '{model_name}' ({e}); "
              f"using whitespace proxy. Counts will UNDERESTIMATE real points.")
        return proxy_token_count, "whitespace_proxy"


def count_points(items, count_fn) -> dict[str, int]:
    """Map item id -> point count (token count of joint premise+hypothesis)."""
    return {it.id: int(count_fn(it.joint_text)) for it in items}


def assess_adequacy(
    counts: dict[str, int],
    threshold: int = THRESHOLD_POINTS,
    coverage: float = COVERAGE_REQUIRED,
) -> dict:
    """Decide per-dimension adequacy from point counts alone."""
    vals = list(counts.values())
    n = len(vals)
    n_meeting = sum(1 for v in vals if v >= threshold)
    fraction = (n_meeting / n) if n else 0.0
    high_dim = "adequate" if (n and fraction >= coverage) else "inconclusive"

    summary = {
        "n_items": n,
        "min": min(vals) if vals else None,
        "median": statistics.median(vals) if vals else None,
        "mean": (sum(vals) / n) if n else None,
        "max": max(vals) if vals else None,
        "n_below_threshold": n - n_meeting,
        "fraction_meeting_threshold": fraction,
    }
    adequacy = {
        "H0": "adequate",
        "H1": "adequate",
        "H2": "adequate",
        "H3": high_dim,
        "H4": high_dim,
    }
    return {"summary": summary, "adequacy": adequacy,
            "threshold_points": threshold, "coverage_required": coverage}


def bootstrap_coverage(
    values,
    threshold: int = THRESHOLD_POINTS,
    n_iter: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for the fraction of items meeting the point threshold.

    Resamples items with replacement and recomputes the coverage fraction each
    time, so the gate decision rests on a stable estimate with an interval
    rather than a single favorable sample. Returns mean and 95% percentile CI.
    """
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    rng = np.random.default_rng(seed)
    fracs = np.array([
        float((rng.choice(arr, size=n, replace=True) >= threshold).mean())
        for _ in range(n_iter)
    ])
    return {
        "mean": float(fracs.mean()),
        "ci_lo": float(np.percentile(fracs, 2.5)),
        "ci_hi": float(np.percentile(fracs, 97.5)),
    }


def run_preflight(cfg: dict, log=print) -> dict:
    corpus_dir = Path(cfg["corpus"]["output_dir"])

    # Natural-language confirmatory set: contradiction + matched neutral control.
    nl_items = []
    for fname in ("contradiction.jsonl", "baseline.jsonl"):
        p = corpus_dir / fname
        if p.exists():
            nl_items.extend(load_corpus(p))
    synth_items = []
    sp = corpus_dir / "synthetic.jsonl"
    if sp.exists():
        synth_items = load_corpus(sp)

    # Base model name + HF token from config.
    base_model = None
    for m in cfg.get("inference", {}).get("models", []):
        if m.get("role") == "base":
            base_model = m.get("name")
            break
    if base_model is None and cfg.get("inference", {}).get("models"):
        base_model = cfg["inference"]["models"][0].get("name")

    from src.utils.config import get_hf_token
    token = get_hf_token(cfg) if base_model else None

    count_fn, tok_label = load_tokenizer(base_model, token) if base_model else (proxy_token_count, "whitespace_proxy")
    log(f"  tokenizer: {tok_label}")

    nl_counts = count_points(nl_items, count_fn)
    nl_assess = assess_adequacy(nl_counts)
    cov = bootstrap_coverage(nl_counts.values())
    label_dist = dict(Counter(it.label for it in nl_items))
    strictly_clean = set(label_dist) <= {"contradiction", "neutral"}
    point = nl_assess["summary"]["fraction_meeting_threshold"]
    point_clears = point >= COVERAGE_REQUIRED
    robust_clears = cov["ci_lo"] >= COVERAGE_REQUIRED

    log(f"  natural-language items: {nl_assess['summary']['n_items']}, "
        f"median points {nl_assess['summary']['median']}, "
        f"{nl_assess['summary']['n_below_threshold']} below {THRESHOLD_POINTS}")
    log(f"  coverage: {point:.0%} point estimate, "
        f"95% CI [{cov['ci_lo']:.0%}, {cov['ci_hi']:.0%}]")
    log(f"  label distribution: {label_dist} "
        f"(strictly three-way clean: {strictly_clean})")
    log(f"  H3/H4 adequacy (NL confirmatory cell): {nl_assess['adequacy']['H3']}")

    report = {
        "date": date.today().isoformat(),
        "tokenizer": tok_label,
        "blinded": "decided from token counts only; no topology computed",
        "decision_record": {
            "coverage_point": point,
            "coverage_ci": [cov["ci_lo"], cov["ci_hi"]],
            "coverage_required": COVERAGE_REQUIRED,
            "point_clears_gate": point_clears,
            "robustly_clears_gate": robust_clears,
            "median_points": nl_assess["summary"]["median"],
            "label_distribution": label_dist,
            "labels_strictly_three_way_clean": strictly_clean,
        },
        "natural_language": {**nl_assess, "per_item": nl_counts},
    }
    if synth_items:
        synth_counts = count_points(synth_items, count_fn)
        report["synthetic"] = {**assess_adequacy(synth_counts), "per_item": synth_counts}

    out_path = corpus_dir / "preflight_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    log(f"  wrote {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    from src.utils.config import load_config
    cfg = load_config(args.config)
    print("point-count preflight (PH-blinded)\n")
    report = run_preflight(cfg)
    dr = report["decision_record"]
    h3 = report["natural_language"]["adequacy"]["H3"]
    print(f"\nDecision record (Phase A gate):")
    print(f"  coverage point {dr['coverage_point']:.0%}, "
          f"95% CI [{dr['coverage_ci'][0]:.0%}, {dr['coverage_ci'][1]:.0%}], "
          f"required {dr['coverage_required']:.0%}")
    print(f"  point clears gate: {dr['point_clears_gate']}; "
          f"robustly clears (lower CI >= 80%): {dr['robustly_clears_gate']}")
    print(f"  labels strictly three-way clean: {dr['labels_strictly_three_way_clean']}")
    print(f"\nResult: H3/H4 are {h3.upper()} for the natural-language confirmatory cell.")
    if h3 == "inconclusive":
        print("Per the protocol, H3/H4 will be reported as inconclusive/unavailable, "
              "not redefined to a lower dimension.")


# ----- Sanity check (run: python -m scripts._preflight_adequacy is N/A; use the file) -----


def _sanity_check() -> None:
    failures: list[str] = []

    def check(name, cond, detail=""):
        marker = "PASS" if cond else "FAIL"
        tail = f": {detail}" if (detail and not cond) else ""
        print(f"  [{marker}] {name}{tail}")
        if not cond:
            failures.append(name)

    # assess_adequacy: 9/10 meet -> 0.9 >= 0.8 -> adequate.
    c = {f"i{j}": (30 if j < 9 else 10) for j in range(10)}
    a = assess_adequacy(c)
    check("9/10 above threshold -> H3 adequate", a["adequacy"]["H3"] == "adequate",
          detail=str(a["summary"]["fraction_meeting_threshold"]))
    check("low dims always adequate", a["adequacy"]["H0"] == "adequate" and a["adequacy"]["H2"] == "adequate")

    # 7/10 -> 0.7 < 0.8 -> inconclusive.
    c2 = {f"i{j}": (30 if j < 7 else 10) for j in range(10)}
    check("7/10 above threshold -> H3 inconclusive",
          assess_adequacy(c2)["adequacy"]["H3"] == "inconclusive")

    # exactly 8/10 = 0.8 -> adequate (boundary, >=).
    c3 = {f"i{j}": (30 if j < 8 else 10) for j in range(10)}
    check("exactly 80% coverage -> adequate (inclusive boundary)",
          assess_adequacy(c3)["adequacy"]["H3"] == "adequate")

    # summary stats correct.
    c4 = {"a": 10, "b": 20, "c": 30}
    s = assess_adequacy(c4)["summary"]
    check("summary median correct", s["median"] == 20)
    check("summary min/max correct", s["min"] == 10 and s["max"] == 30)
    check("n_below_threshold correct (25): two below", s["n_below_threshold"] == 2,
          detail=str(s["n_below_threshold"]))

    # empty input handled.
    e = assess_adequacy({})
    check("empty input -> inconclusive, n_items 0",
          e["adequacy"]["H3"] == "inconclusive" and e["summary"]["n_items"] == 0)

    # count_points with a stand-in tokenizer (word count).
    class FakeItem:
        def __init__(self, id, text): self.id = id; self.joint_text = text
    items = [FakeItem("x", "one two three"), FakeItem("y", "a b c d e")]
    counts = count_points(items, lambda t: len(t.split()))
    check("count_points maps id -> token count", counts == {"x": 3, "y": 5}, detail=str(counts))

    # proxy counter is a sane integer.
    check("proxy_token_count returns int >= word count",
          proxy_token_count("hello world, this is text") == 5,
          detail=str(proxy_token_count("hello world, this is text")))

    # bootstrap_coverage: all above threshold -> coverage 1.0, tight CI.
    allhi = bootstrap_coverage([30, 40, 50, 60], threshold=25)
    check("coverage all-above -> mean 1.0", allhi["mean"] == 1.0 and allhi["ci_lo"] == 1.0)
    # all below -> 0.0.
    alllo = bootstrap_coverage([10, 12, 5], threshold=25)
    check("coverage all-below -> mean 0.0", alllo["mean"] == 0.0 and alllo["ci_hi"] == 0.0)
    # mixed ~50/50 -> mean near 0.5, CI strictly inside (0,1).
    mixed = bootstrap_coverage([30] * 50 + [10] * 50, threshold=25)
    check("coverage mixed -> mean near 0.5",
          0.4 < mixed["mean"] < 0.6, detail=str(mixed["mean"]))
    check("coverage mixed -> CI lower bound below 0.8 (not a clean clear)",
          mixed["ci_lo"] < 0.8)
    # empty -> zeros.
    check("coverage empty -> zeros", bootstrap_coverage([]) == {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {failures}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    # If --config is given, run the real preflight; otherwise run the sanity check.
    if "--config" in sys.argv:
        main()
    else:
        print("preflight adequacy sanity check\n")
        _sanity_check()
