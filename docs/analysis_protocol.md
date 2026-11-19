# Murmur Analysis Protocol (preregistration)

**Status:** Preregistered. Written 2026-06-30, before any cloud spend and before any real model output exists.
**Repo state at writing:** commit `e53a4cb` (analysis trio shipped, session 1 complete).
**Scope:** the dimensional-specificity experiment, both arms, both stages.

This document locks the analysis decisions in advance. Anything changed after this date is recorded as a dated entry in the Amendment Log at the bottom, with a reason. No silent edits. The point of this file is to make it impossible to retrofit the theory to whatever the data turns out to be.

---

## 0. Standing principle: necessity over baselines (anti-tautology rule)

This applies to this experiment and to every Murmur study that claims the geometry or topology of a representation carries information.

Stated plainly: a topological result counts only if a simpler, standard, position-readable method cannot already produce it on the same data. Persistent homology reads the *shape* of a point cloud; cosine distance, linear probes, model logits, and PCA read the *positions* of the points. The entire Murmur thesis is that shape carries information that position does not. So any positive topological result must be shown to exceed position-only baselines on the identical items, or it is reported as not supporting the topological claim, however clean the topology looks on its own. If a logistic regression on the raw embeddings separates the classes as well as the H_k void does, the topology is decoration, not evidence.

A theory stops being a tautology when it can lose. This rule is one of the ways every Murmur claim is built to be able to lose.

Scope: when a study's claim does not depend on shape-beyond-position (a feasibility or descriptive measurement, such as the Phase A point-count preflight), this gate is not required. Whenever the claim is that the topology is doing the work, it is required.

---

## 1. Confirmatory target and analysis hierarchy

There is exactly one confirmatory claim. Everything else is secondary and labeled as such, so a low-powered subtype result cannot break an otherwise clean primary result.

**Primary (confirmatory) claim:**

> Per-token arm, Llama-3.1-8B **base**, pooled **natural-language** contradiction set, dimensional dominance within H2-H4.

The Holm-Bonferroni family (Section 5.1) covers only the dimensions inside this one cell. Nothing else enters the confirmatory correction.

**Window status (locked 2026-06-30).** The full H2-H4 window is confirmed testable on the confirmatory corpus. The window was held as the sole unlocked variable until the Phase A exploratory preflight reported; that preflight cleared the Section 8 adequacy gate on ANLI, so the window is now locked to H2-H4. See Amendment Log.

**Everything else, pre-specified and separately reported:**

- **Replication:** Llama-3.1-8B-Instruct, same arm, same statistic.
- **Secondary architecture test:** the attention-graph flag-complex arm.
- **Robustness:** the six Wu-2022 contradiction categories analyzed individually.
- **Validity / controls:** synthetic items, including the controlled coherent-edit specificity check (Section 4.3).
- **Adjacent-dimension analyses:** H1 as candidate (Section 6.1) and the Jiang-aligned H0 baseline (Section 6.2).

A positive in a secondary analysis without a positive primary is reported as a secondary finding, never relabeled as the headline.

---

## 2. Inferential unit

The matched **pair** is the unit of analysis and the unit of resampling. Every contradiction item `c` is paired with one **coherent-edit control** item `e`, matched per Section 7. The coherent-edit control is the NLI **neutral** item, a coherent semantic change to the premise that is neither entailed nor contradictory. This is what makes the primary contrast contradiction-specific rather than "any semantic change moves the cloud."

Nothing in the pipeline resamples, permutes, or aggregates in a way that severs `c` from its `e`.

**Source-family independence.** Each source text or generation template contributes at most one independent pair to the confirmatory set. Where one source spawns multiple variants (synthetic especially), the resampling unit is the source-family, not the variant, or precision is overstated. The bootstrap (Section 4.4) resamples over an explicit unit id: the pair id on the natural-language confirmatory set (where pairs come from distinct sources), the source-family id on synthetic robustness analyses.

A "cell" is one combination of (arm, model, dimension `k`, contradiction-type). Tests are computed per cell; only the primary cell's dimensions enter the confirmatory correction.

---

## 3. Dimensions computed

H0 through H4 are computed for every item in both arms. No dimension is excluded at the topology stage. The primary prediction lives in {2, 3, 4}; H1 and H0 are analyzed as adjacent dimensions per Section 6.

---

## 4. Primary statistic

### 4.1 Per-item scalar: max persistence

For item `x` and dimension `k`, with H_k persistence diagram as a set of (birth, death) pairs:

```
M_k(x) = max over features of (death - birth)      # longest-lived H_k feature
M_k(x) = 0 if the H_k diagram is empty
```

This is the single most persistent void at dimension `k`. Note `M_k` equals **twice** the bottleneck distance from the H_k diagram to the empty diagram, since a point's distance to the diagonal is (death - birth)/2. The factor is constant across items and does not affect the statistic; it is stated so the relationship to the bottleneck machinery is correct on the record.

### 4.2 Normalization (locked)

Per-token arm: divide all (birth, death) values by the item's point-cloud diameter (max pairwise Euclidean distance among the item's token embeddings) before computing `M_k`. This makes `M_k` a dimensionless fraction of the cloud's scale and comparable across items with different embedding norms. The H0-longest-bar alternative is rejected because it would couple the normalizer to the H0 statistic tested separately in Section 6.2.

Attention arm: the flag-complex filtration runs over density thresholds already in [0, 1], so `M_k` is dimensionless. No further normalization.

### 4.3 Per-pair contrast and effect size

For matched pair `p = (c_p, e_p)`:

```
Delta_k(p) = M_k(c_p) - M_k(e_p)     # signed, contradiction minus coherent-edit control
```

Primary effect size, paired (one-sample) Cohen's d_z over pairs:

```
d_z(k) = mean_p[ Delta_k(p) ] / sd_p[ Delta_k(p) ]
```

A positive `d_z(k)` means contradiction items carry a more persistent H_k void than their matched coherent-edit controls. Because the comparison partner is a coherent edit rather than a no-change baseline, a positive result is contradiction-specific by construction.

### 4.4 Resampling

- **Bootstrap (interval estimate):** resample units with replacement (pair id on the confirmatory set, source-family id on synthetic), 10,000 iterations, recompute `d_z(k)` each iteration, report the 95% percentile CI. The resampling unit is never the individual item.
- **Permutation (inferential anchor):** 10,000 iterations. In each, independently per pair, swap the contradiction and coherent-edit label with probability 0.5, recompute `mean_p[ Delta_k(p) ]`. Two-sided p = fraction of permuted |mean| at least the observed |mean|. The swap is within-pair, which preserves the matching under the null.

---

## 5. Primary hypothesis test: H2-H4 dominance

Computed on the primary cell only (Section 1). Declare a single dominant dimension `k*` in {2, 3, 4} only if all three hold:

1. **Margin:** `d_z(k*)` exceeds the second-largest `d_z` in {2, 3, 4} by at least **0.2** d_z units.
2. **Floor:** `d_z(k*) >= 0.5`, a medium effect.
3. **Significance:** the permutation test for `k*` rejects under the family-wise correction in 5.1.

### 5.1 Multiple comparisons

Holm-Bonferroni across the permutation tests for the dimensions in the primary cell, family-wise **alpha = 0.05**. The family is fixed here at preregistration. Secondary, replication, robustness, and validity analyses carry their own separately reported corrections and never enter the primary family.

### 5.2 Falsification

The primary hypothesis is falsified if any of:

- No dimension in {2, 3, 4} clears both margin and floor.
- The permutation test for the leading dimension fails to reject under Holm-Bonferroni.
- Signal appears in synthetic but not natural-language items (handled as a validity flag, Section 4.3 / 6).
- Any control in Section 7 reproduces the contradiction-specific separation.
- The topological separation at k* does not exceed the position-only baselines (Section 14) on the identical pairs. If the effect is fully position-readable, the geometry adds nothing and the topological claim is unsupported, even where separation exists.
- The dominant dimension k* does not replicate on a different-lineage transfer corpus (Section 15). Dimensional specificity that holds only on the corpus it was fixed on is unsupported as a general claim.

A falsification of dimensional specificity is publishable. It localizes where the contradiction signal actually lives.

---

## 6. Adjacent-dimension analyses (pre-specified, separate)

### 6.1 H1 as candidate

`d_z(1)` is computed and reported the same way as the primary dimensions, openly, not as a buried control. Three preregistered readings:

1. **H1 separation stronger than the H2-H4 winner.** Falsifies dimensional-specificity as framed, does not falsify the architecture; the trigger could operate on H1. The paper would then have to specify what distinguishes a contradiction loop from a legitimate loop.
2. **H1 separates, but a single H2-H4 dimension is stronger.** Strong positive, H1 noted as related secondary structure.
3. **H1 does not separate, a single H2-H4 dimension does.** Cleanest positive; the exclusion-by-argument was correct.

### 6.2 Jiang-aligned H0 baseline

`d_z(0)` on `M_0`. This is **not** a strict replication of Jiang et al. (2026), who concatenate three H0 descriptors (mean finite lifetime, normalized lifetime entropy, largest-lifetime concentration) across layers. Our single max-persistence scalar is a Jiang-aligned baseline. Reported as baseline-and-extension, never as a primary Murmur result. A full Jiang replication using their three descriptors is optional follow-up, out of scope here.

---

## 7. Corpus construction and controls

**Primary comparison partner (coherent-edit control):** NLI **neutral** items, matched to their contradiction item on:

- Length within 15% (token count).
- Vocabulary overlap (Jaccard) at least 0.30.
- Syntactic-complexity within 20%.

**Low-change reference:** NLI **entailment** items, matched the same way. Reported as a secondary contrast (near-zero semantic change), not the primary partner.

**Type balance and quality:** the six Wu-2022 categories, balanced; natural-language items at annotator agreement at least 4/5.

**Synthetic items and the controlled coherent-edit check (Section 4.3 validity):** for each synthetic contradiction (single operator flip, single numeric change, single negation), generate a magnitude-matched coherent edit (synonym swap, plausible non-contradictory numeric change, coherence-preserving paraphrase). The synthetic contradiction-versus-coherent-edit contrast is the controlled-magnitude specificity check, complementing the natural-language neutral control.

**Additional controls (each must fail to reproduce the contradiction-specific separation):**

- Label permutation.
- Semantic-preserving paraphrase invariance.
- Token-order shuffle.
- Matched random Gaussian point clouds at the same point count.

The pairing defined here is preserved through every bootstrap resample and every permutation iteration in Section 4.4.

---

## 8. Point-cloud adequacy gate

Max persistence is driven by a single feature, so a lone spurious long-lived bar in a sparse cloud can drive `M_k` on its own. This gate protects the primary statistic, and it runs **before** any topology is computed so it cannot become a post-hoc rewrite of the hypothesis.

- **PH-blinded engineering preflight.** Run the tokenizer over the full corpus and count points per item. Decide dimension adequacy from these counts alone, with no persistent homology and no separation result visible.
- **Per-item threshold:** any item with fewer than **25 points** is flagged.
- **Coverage rule:** a dimension is adequate in a cell only if at least **80%** of that cell's items meet the point-count threshold. A median alone can hide an unusable tail, so the median is not used as the gate.
- **No downward redefinition.** If H3 or H4 fail the adequacy gate, the high-dimensional test for that cell is reported as **inconclusive / unavailable**, never relabeled as success at a lower dimension. Lower-dimensional results remain reportable as secondary. The primary window is not contracted after data.
- **Phase A result (2026-06-30, exploratory preflight).** The PH-blinded preflight (tokenizer-only, no topology) was run on candidate sources. SNLI cleared at 53% coverage and MultiNLI at 67%, both short of the 80% gate. ANLI cleared at 100% coverage (95% CI [100%, 100%], median 93 points, labels strictly three-way clean). The gate is therefore met on the ANLI confirmatory corpus and H3/H4 are testable. This preflight is exploratory; it fixes the testable window before any topology or separation result is seen.

---

## 9. Sampling, allocation, quarantine

- **Stage 1 size:** exactly **150 matched pairs** on the per-token arm, base model. "Instance" means matched pair throughout.
- **Allocation:** the natural-language set is drawn from **ANLI** (paragraph-length Wikipedia/HotpotQA premises, clean three-way labels), **pooled**. Per-type balance across the six Wu-2022 categories is **deferred**: no Wu-2022 type classifier is built, the NLI loaders do not assign a contradiction type, and the corpus builder does not enforce type balance. The confirmatory claim is on the pooled set (Section 1), which needs no type labels; the per-type robustness analysis (Section 1, secondary) is deferred pending that classifier and is reported as not-yet-run rather than null. DocNLI is excluded: its binary labels map not-entailment to "contradiction" and it has no neutral class, so it cannot supply clean three-way data (Amendment Log 2026-06-30). SNLI and MultiNLI are not used for the confirmatory set because they fail the Section 8 adequacy gate (53% and 67% coverage); they remain available for short-context secondary analyses.
- **Quarantine:** Stage 1 pairs are excluded from Stage 2. No item appears in both stages.
- **Stage 2 size:** the largest pair count fundable under the $500 cap per `docs/budget.md`, consistent with v2's roughly 4,500 per-token and 900 attention-graph item figures. The exact Stage 2 count is locked against `docs/budget.md` before Stage 2 begins and recorded in the Amendment Log. It does not need to be fixed before Stage 1 runs.
- **Incomplete runs:** if a budget or extraction-failure stop fires before the planned n is reached, the result is descriptive / incomplete and is **not** eligible for the primary confirmatory test.

---

## 10. Representation pipeline lock

The statistics above are locked; the representation choices are where a favorable result could otherwise be fished by changing the geometry before it reaches the protocol. The following are frozen before cloud spend. As of 2026-06-30 all items are locked; no TODO remains open.

| Choice | Value |
|---|---|
| Model (primary) | Llama-3.1-8B base |
| Model (replication) | Llama-3.1-8B-Instruct |
| Exact HF model revision (commit SHA) | `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` (meta-llama/Llama-3.1-8B, pinned 2026-06-30) |
| Tokenizer | Llama-3.1 tokenizer shipped with the model |
| Quantization | 4-bit NF4 via bitsandbytes |
| Compute dtype / precision | bfloat16 compute dtype over the 4-bit weights |
| Per-token substrate | post-block hidden states (residual stream) |
| Layer extracted | **layer 24 of 32** (`hidden_states[24]`, residual stream after block 24); a single layer, committed blind. A layer sweep is reported as exploratory only, never used to pick the confirmatory layer after seeing results. |
| Special tokens in point cloud | exclude BOS/EOS/pad; keep all content and punctuation subword tokens |
| Distance metric | Euclidean |
| Centering / normalization | per-item diameter normalization (Section 4.2) |
| VR construction | ripser; max_dim per config; sparse_filtration_threshold = 500 |
| Coefficient field | Z/2Z |
| Flag complex | networkx clique enumeration to simplicial complex |
| Zigzag | dionysus 2.2.3 |
| Attention density sweep | 5% to 50% |
| Attention head/layer aggregation | mean over heads within each layer; layers 22-32 |
| Attention directionality / symmetrization | symmetrize the directional attention matrix as (A + Aᵀ)/2 before building the graph |
| Inference determinism | seed 42 everywhere; deterministic mode; no sampling (hidden states are read, not generated) |
| Software versions | pinned via `requirements.txt` freeze at dry run; Python 3.11.13, torch, ripser, persim, networkx, dionysus 2.2.3, numpy |

---

## 11. Stage 1 to Stage 2 transition

Stage 1 (150 pairs, H0-H2 only) cannot see the top of the primary window. Its job is to confirm there is contradiction-versus-coherent-edit signal worth paying for the full sweep.

Greenlight Stage 2 only if all hold:

- At least one of {H1, H2} shows `d_z >= 0.30` (deliberately low: Stage 1 is underpowered and this is a go / no-go, not a confirmation).
- That dimension's permutation p < 0.05, uncorrected (Stage 1 is exploratory; Holm-Bonferroni applies at Stage 2).
- The Section 7 controls show nothing comparable at Stage 1 scale.

Flat across H0-H2 means do not spend the Stage 2 budget. Write up the negative.

---

## 12. Stage 2 stop conditions

- **No interim peeking on the primary statistic.** Run to the planned, budget-locked n, then test once. Optional stopping inflates the error rate and is forbidden.
- **Hard operational stops only:** budget cap of $500 reached, or unrecoverable extraction failure rate above 10%. Either makes the run incomplete per Section 9.
- Computational blowups on the attention arm trigger the fallback in Section 13, not a stop.

---

## 13. Attention-arm fallback

The attention arm is secondary by design. If flag-complex H2+ is infeasible (per-item simplex count or wall-clock past the feasibility threshold measured at dry run, or out-of-memory), retreat the attention arm to H0/H1 only and keep the per-token arm as primary at the full H2-H4 window. The attention arm failing never blocks the primary result.

---

## 14. Baselines and necessity (confirmatory gate)

This section operationalizes Section 0 for this experiment. The primary topological result is confirmed only if it beats position-only baselines on the identical matched pairs. Every baseline is computed on the same per-token embeddings and the same contradiction-versus-coherent-edit pairs as the primary statistic, so the only thing that differs between them is whether the method reads shape or position.

**Position-only baselines (identical pairs):**

- **Cosine / Euclidean separation:** distance between the contradiction item's mean (or pooled) embedding and its coherent-edit control's, per pair.
- **Linear probe:** L2-regularized logistic regression discriminating contradiction from coherent-edit on the pooled / mean embedding, evaluated with source-family-grouped cross-validation so no source leaks across folds.
- **Model-internal signal:** the base model's own entailment / contradiction readout where available (NLI head logits, or a zero-shot prompt score), as a "the model already knows" baseline.
- **Linear-subspace separation:** discrimination in the top principal components (PCA) of the embeddings, to rule out that a few linear directions already carry the effect.

**Common metric.** The topological scalar M_k* and each baseline are scored by their discrimination of contradiction versus coherent-edit on the identical pairs, reported as AUC and as a paired separation effect size, bootstrapped over the same units as Section 4.4.

**Necessity criterion (confirmatory).** The topological discrimination at k* must exceed the best position-only baseline by at least **0.05 AUC** (default; a topological effect that merely ties a linear probe is not evidence that shape matters). The stronger incremental-validity form is reported alongside: a nested comparison of (baseline features) versus (baseline features + M_k*), where the topological feature must add discrimination beyond the baselines. The head-to-head margin is the gate; the incremental test is reported with it.

---

## 15. Cross-corpus transfer

Dimensional specificity is a claim about contradiction, not about ANLI. The dominant dimension k* and the thresholds (margin, floor) are fixed on the ANLI confirmatory corpus and then applied **without adjustment** to a second clean three-way source.

- **Different-lineage transfer (primary).** A clean three-way source of different dataset lineage than ANLI that clears the Section 8 adequacy gate. Candidates: FEVER-NLI, a long-premise subset of MultiNLI, or ConTRoL; each must pass its own PH-blinded preflight (exploratory) before use. The same k* must dominate, with margin and floor still met and the Section 14 necessity gate still passed.
- **Same-lineage stability (secondary, weaker).** The held-out ANLI dev / test rounds, quarantined from the confirmatory train rounds, as a stability check on k* within lineage.
- If no different-lineage source clears the adequacy gate within budget, the different-lineage transfer is reported as **unavailable** (a boundary condition on available data), and the claim is qualified accordingly rather than asserted as transfer-confirmed.

---

## 16. Amendment log

*(empty at creation; every post-hoc change recorded below with date and reason)*

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-06-30 | 1, 8, 9 | DocNLI removed from corpus sources; ANLI adopted as the primary natural-language source; the full H2-H4 confirmatory window locked as testable. | Phase A exploratory preflight (PH-blinded, tokenizer-only, no topology computed). DocNLI maps not-entailment to "contradiction" and has no neutral class, so it cannot supply clean three-way data; removed. SNLI and MultiNLI cleared the 25-point coverage gate at only 53% and 67%. ANLI (paragraph-length premises, clean three-way labels) cleared at 100% coverage, 95% CI [100%, 100%], median 93 points, strictly three-way clean, robustly past the 80% gate. The window was held as the sole unlocked variable until this preflight; it is now locked to H2-H4. Phase A is exploratory; the later run on the ANLI corpus is the confirmatory run. |
| 2026-06-30 | 0, 5.2, 14, 15 | Added the necessity-over-baselines standing principle (Section 0), a Baselines and necessity confirmatory gate (Section 14), a cross-corpus transfer test (Section 15), and two matching falsification clauses (Section 5.2). Amendment log renumbered to Section 16. | Anti-tautology guardrail. A topological result must beat position-only baselines on the identical data and replicate across corpora, or the topological claim is not supported. Added before any data, so it only hardens the test, never loosens it. |
| 2026-06-30 | 10 | Representation lock completed: pinned model revision SHA d04e592bb4f6aa9cfee91e2e20afa771667e1d4b, bfloat16 compute over 4-bit NF4, per-token layer 24 of 32, content+punctuation tokens, Z/2Z, attention head-mean with (A + Aᵀ)/2 symmetrization, seed 42 / deterministic. Layer 24 wired into the extraction code and run configs. | All Section 10 TODOs closed before the dry run, per protocol. Note: the extraction code had silently defaulted to the final layer (hidden_states[-1]); the lock corrected it to the committed layer 24, a discrepancy the lock step caught before any cloud spend. |
| 2026-06-30 | 9 | Confirmatory natural-language set is pooled ANLI; per-type (Wu-2022) balance and per-type robustness deferred to secondary, pending a type classifier that is not yet built. | The classifier does not exist, the loaders assign no contradiction type, and the builder does not enforce type balance, so the prior "balanced 25/type" allocation was never realized. The confirmatory claim is pooled (Section 1) and is unaffected. This narrows what is claimed-as-implemented; it does not loosen the confirmatory test. |
| 2026-06-30 | 10 (code verification) | Verified the Section 10 values are honored in code, not only in the table. Found and fixed: the 4-bit compute dtype was torch.float16 in the loader (lock says bfloat16); the model revision was not pinned at load (now threaded from config to both the tokenizer and model `from_pretrained`); the ripser coefficient field was implicit (now `coeff=2` explicit). Confirmed already correct: attention head-mean and (A + Aᵀ)/2 symmetrization in `threshold_attention_to_graph`. | Same class of issue as the layer-24 default. One value was actively wrong, not merely unwired: the dtype was float16, not the locked bfloat16, so this pass changed behavior. Closes the representation lock in code before the dry run. |