# Dimensional specificity of contradiction in language-model representations: a preregistered test with position-only baselines

**Wayne Olson**

*Preprint. Preregistered study: the analysis protocol summarized in Methods was written and dated before any model output existed, and is released with this paper.*

## Abstract

Language models produce fluent output under semantic conditions that should reduce confidence, including contradiction between premises. Recent work reports that such conditions leave topological signal in model internals, but those reports credit the topology by discriminative performance alone, without testing whether a position-only method reads the same signal from the same data. This paper reports a preregistered test of a specific claim: that processing a contradiction produces a persistent-homology signature concentrated at a single homological dimension within H2–H4 of the token-level representation, exceeding matched coherent-edit controls and surviving a necessity gate that requires the topological statistic to beat cosine, linear-probe, model-logit, and PCA baselines by at least 0.05 AUC on identical pairs. The statistic (diameter-normalized maximum persistence, paired Cohen's d_z), the representation (Llama-3.1-8B base, pinned revision, layer 24 of 32), the corpus adequacy gate, and all decision thresholds were locked before data collection. An exploratory first stage (150 ANLI contradiction–neutral pairs, 86 surviving matching, H0–H2) returned a null: no dimension approached the preregistered go/no-go criterion (largest |d_z| = 0.06, all permutation p > 0.59, Holm-adjusted p = 1.0), and the confirmatory stage was therefore not run. The null is reported under the protocol's falsification structure together with its principal confound, a single extraction depth that interpretability evidence places above the layers where relational features concentrate. A positive-control study, whose design and detection criteria are fixed in advance in this paper, is the pre-committed next test and determines whether the instrument or the configuration explains the result.

## 1. Introduction

A language model can generate text that is locally coherent while the conditions for trusting that text have failed. Documented failure modes include confabulation of unsupported content [37, 2], failure to register contradiction between premises [10], reliance on stale or conflicting sources, and confident completion over missing information. These phenomena are typically studied separately. They share one structural property: the surface form of the output remains intact while the semantic relations that would license it do not. This paper uses the term semantic discord for that shared property. The term is introduced here; the component phenomena are established in prior literature under their individual names.

Methods for detecting such conditions divide by where they look. The dominant line operates downstream, on generated outputs: semantic entropy clusters sampled answers by meaning and measures dispersion over clusters [1, 2]; subsequent work replaces the equivalence relation with richer structure, including entailment graphs [4], implication and incompatibility graphs [3], and structural entropy over semantic graphs [5]. These methods require multiple generations per query and pairwise inference among them, with a documented five-to-tenfold inference-cost overhead [35]. A second line operates upstream, on internal representations, and reads them either linearly, through probes and steering directions [23, 24, 25, 26, 27], or structurally, through topological data analysis [12, 13, 14, 15, 22]. The topological sub-line is motivated in part by evidence that token-embedding spaces are not smooth manifolds: neighborhoods exhibit multimodal local dimension and singularities that persist as context grows [20], which limits the applicability of smooth-geometric methods and makes coordinate-free multiscale summaries the natural alternative [6, 7].

The topological reports to date share a methodological gap. Adversarial perturbation compresses latent-space topology consistently across model scales [12, 21]; hallucinated generations are separable from factual ones by zigzag persistence over layer-wise attention graphs [22]; contradiction in text was classified with topological features over static word embeddings [10]. In each case the topological signal is credited by its discriminative performance. None of these studies tests whether a linear read of the same representations discriminates as well. The concern is not hypothetical. In representation probing, Hewitt and Liang showed that probe accuracy without a control baseline is uninterpretable, because the probe may learn the task rather than read the representation [36]; the field adopted control tasks in response. An equivalent discipline is absent from the topological literature. It is needed for the same reason, and for one more: a linear probe on hidden states is now a published, near-zero-cost reliability instrument in its own right, recovering much of the semantic-entropy signal from a single forward pass [35]. A topological statistic that fails to exceed that baseline is not merely under-supported; it is dominated by a cheaper method.

This paper contributes a preregistered protocol that closes the gap for one concrete claim, and reports its first application. The claim under test is scoped precisely. Murmur, the research program this test belongs to, is a continuous-state architecture in which memory weighting, semantic tension, and contradiction are represented in a model's evolving internal state; its operator definitions are covered by a provisional filing and are not used in this paper. Within that program, topology is one candidate instrument for reading representation-level structure. Within topology, contradiction is one tractable instance of discord. Within contradiction, the tested prediction is dimensional specificity: a signature concentrated at a single homological dimension, not diffuse structure. And within that prediction, the present study is a single exploratory stage at a single extraction depth. The results below are interpreted at that level and no higher.

## 2. The prediction

Persistent homology summarizes a point cloud by the birth and death scales of its topological features across a filtration: H0 features are connected components, H1 features are loops, and Hk features for k ≥ 2 are k-dimensional voids [6, 19]. The persistence of a feature, death minus birth, is stable under perturbation of the underlying points [8]. The prediction tested here is:

> When a language model processes a contradiction, the token-level point cloud of its intermediate representation carries a persistent Hk feature, for a single k in {2, 3, 4}, that is systematically more persistent than in a matched coherent control, by a margin that (i) singles out one dimension, (ii) reaches a medium paired effect size, and (iii) exceeds what position-only methods extract from the same embeddings.

Clause (iii) is the necessity condition and is treated as constitutive, not supplementary: a topological separation that a linear probe matches is reported as unsupported regardless of its own magnitude. Dimensional specificity, clauses (i) and (ii), distinguishes the prediction from the weaker claim that contradiction perturbs representation geometry somehow. Dimension-specific topological response to semantic events has precedent at the discourse level, where H0 and H1 entropies carry signal on distinct timescales [34]; the prediction extends specificity to higher dimensions inside a single forward pass. The window {2, 3, 4} excludes H0 and H1 by argument rather than by testing: H0 structure tracks clustering, which ordinary semantic difference already moves [14], and H1 loops arise from legitimate recurrence in text. Both are computed and reported as adjacent-dimension analyses; a stronger H1 than H2–H4 result is a preregistered falsification of specificity as framed.

## 3. Methods

The full protocol, with its amendment log, is released alongside this paper. It was written on 2026-06-30, before any cloud inference. Its dated commit history is in the project's private repository and is open to inspection on request. All thresholds below are quoted from it.

### 3.1 Corpus and matched pairs

The confirmatory corpus is drawn from ANLI, pooled across contradiction types. Corpus selection was governed by a persistence-blinded adequacy gate run before any topology was computed: the tokenizer was applied to candidate sources, points per item were counted, and a dimension was deemed testable in a cell only if at least 80% of items carried at least 25 points, a threshold protecting a maximum-persistence statistic from single-bar artifacts in sparse clouds. SNLI reached 53% coverage and MultiNLI 67%, and both were excluded from the confirmatory set. ANLI reached 100% coverage (95% CI [100%, 100%], median 93 points per item) and locked the testable window at H2–H4. DocNLI was excluded because its binary labels collapse neutral into contradiction and cannot supply clean three-way data.

The inferential unit is the matched pair. Each contradiction item is paired with a coherent-edit control: the NLI neutral item, a coherent semantic change that is neither entailed nor contradictory, matched on token length within 15%, Jaccard vocabulary overlap of at least 0.30, and syntactic complexity within 20%. The neutral control makes the contrast contradiction-specific by construction; a no-change or entailment baseline would confound contradiction with semantic change per se. Entailment items matched by the same criteria are retained as a secondary low-change reference. Pairing is preserved through every resampling operation. Stage 1 comprised 150 sampled pairs, of which 86 satisfied the matching criteria and entered analysis; Stage 1 items are quarantined from any later confirmatory stage. The protocol specified 150 matched pairs for Stage 1; the 86 analyzed fall short of that preregistered size.

### 3.2 Representation extraction

All representation choices were frozen in the protocol before data collection. Model: Llama-3.1-8B base (replication target: the Instruct variant), pinned to Hugging Face revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b, loaded in 4-bit NF4 quantization with bfloat16 compute. Extraction reads the residual stream after block 24 of 32, a single depth committed blind; a layer sweep is designated exploratory and cannot be used to select the confirmatory depth after results are seen. The point cloud for an item is its content and punctuation subword tokens (BOS, EOS, and padding excluded) at that depth, under Euclidean distance. Inference is deterministic, seed 42; hidden states are read, not generated. During pre-run verification of this lock, two discrepancies between protocol and code were found and corrected: extraction defaulted to the final layer rather than the committed layer 24, and the quantization compute dtype was float16 rather than the locked bfloat16. Both were fixed before any cloud inference, and both corrections are recorded in the protocol's amendment log.

### 3.3 Topological computation

Vietoris–Rips persistence is computed per item with ripser [9] over Z/2Z, dimensions H0 through H4 (Stage 1 configuration: H0–H2). A secondary attention-graph arm builds flag complexes from head-averaged attention matrices, symmetrized as (A + Aᵀ)/2, over density thresholds from 5% to 50% of edges, with zigzag persistence [15] across layers 22–32; this arm is preregistered as secondary and its feasibility never gates the primary result. Stage 1 ran the per-token arm only.

### 3.4 Statistic and inference

The per-item scalar at dimension k is maximum persistence, M_k = max over finite Hk features of (death − birth), zero for an empty diagram, with all birth and death values divided by the item's point-cloud diameter beforehand so that M_k is a dimensionless fraction of the cloud's scale. The per-pair contrast is Δ_k(p) = M_k(contradiction) − M_k(control), and the effect size is paired Cohen's d_z over pairs. Interval estimates use a 10,000-iteration percentile bootstrap resampling pairs (or source families, where one source contributes multiple variants); p-values use a 10,000-iteration within-pair sign-flip permutation of the mean contrast, two-sided. Confirmatory dominance at a single k* in {2, 3, 4} requires all of: a margin of at least 0.2 d_z units over the runner-up dimension, a floor of d_z ≥ 0.5, and permutation significance under Holm–Bonferroni at family-wise α = 0.05, the family fixed at preregistration as the primary cell's in-window dimensions. Secondary analyses (Instruct replication, attention arm, per-type robustness, H0 and H1) carry separate corrections and cannot enter the primary family.

### 3.5 Position-only baselines and the necessity gate

Four baselines are computed on the identical embeddings and identical pairs: cosine and Euclidean separation of pooled embeddings; an L2-regularized logistic probe with source-family-grouped cross-validation; the model's own zero-shot entailment readout; and discrimination in the top principal components. The topological statistic and each baseline are scored as AUC for contradiction versus control. The necessity criterion requires the topological discrimination at k* to exceed the best baseline by at least 0.05 AUC, with an incremental nested comparison (baselines versus baselines plus M_k*) reported alongside. Failure of this gate falsifies the topological claim even where the topological separation is itself significant.

### 3.6 Staging and falsification

Stage 1 (150 pairs, H0–H2) is an exploratory go/no-go: Stage 2, the full H2–H4 confirmatory run to a budget-locked n with no interim analysis, proceeds only if some dimension in {H1, H2} reaches d_z ≥ 0.30 with uncorrected permutation p < 0.05 and flat controls. The protocol enumerates falsification conditions in advance, including: no dimension clearing margin and floor; failure under Holm; signal in synthetic but not natural items; any control (label permutation, paraphrase invariance, token-order shuffle, matched Gaussian clouds) reproducing the separation; failure of the necessity gate; and failure of k* to transfer, at fixed thresholds, to a different-lineage corpus that passes its own adequacy preflight.

## 4. Results

### 4.1 Pipeline verification

The pipeline was validated end to end before measurement. On planted geometry, contradiction items constructed as spherical shells against control blobs, it recovered the separation at H2 with d_z ≈ 0.53, confirming signal direction and plumbing. On a 48-item noise-grade dry run with real GPU inference, it correctly reported no confirmation. The representation-lock verification results (Section 3.2) are part of this record: two silent code deviations from the preregistered configuration were caught by checking the lock against running code rather than against documentation.

### 4.2 Stage 1

Stage 1 executed on a single RTX A5000: 284 items embedded (150 contradictions, matched controls, and synthetic items), 86 matched pairs analyzed at layer 24, H0–H2, diameter-normalized, per-token arm. The original results artifact was lost to ephemeral pod storage before it was committed; the pipeline was re-executed under the locked configuration (seed 42, pinned model revision) on an RTX 4090, reproducing the two July point estimates that survived (H0 and H1) to the second decimal, with third-decimal differences attributed to hardware-dependent floating-point reduction order. The regenerated artifact is the canonical record and the one released.

No dimension approached the go/no-go criterion. H0: d_z = −0.057, 95% CI [−0.251, +0.163], permutation p = 0.603. H1: d_z = −0.014, CI [−0.236, +0.199], p = 0.895. H2, the strongest in-window dimension by point estimate: d_z = +0.029, CI [−0.196, +0.229], p = 0.787, satisfying neither the margin nor the floor criterion. Holm-adjusted p-values across the tested window were 1.0. H0 and H1 carried signs opposite to the prediction; H2's sign matched it, at negligible magnitude and chance-level significance. Under Section 3.6, Stage 2 was not run, and the confirmatory necessity comparison, which gates a positive result, was therefore not triggered; the baseline implementations were validated on synthetic data but have not been applied to a confirmatory run.

The intervals also bound what the data exclude. The preregistered dominance margin was 0.2 d_z. Effects beyond |d_z| of roughly 0.23 to 0.25 are excluded at the 95% level in every measured dimension (extreme interval endpoints −0.251, −0.236, +0.229), but each interval crosses ±0.2 on at least one side, so equivalence at the preregistered margin is not established: the study rules out medium effects and is silent on the band between 0.20 and roughly 0.25. Stage 1 was, by design, powered only against medium effects (Section 5.5).

The result is a failure to detect under one configuration, not a demonstration of absence, and the protocol constrains which further analyses may be run before that distinction is resolved (Section 5.3).

## 5. Discussion

### 5.1 What the null does and does not establish

The tested configuration involved one extraction depth, one statistic, one pooled corpus, and no positive control establishing the instrument's sensitivity to any contradiction signal. A null under these conditions is consistent with two states: natural ANLI contradictions carry no diameter-normalized maximum-persistence signature at layer 24, or the instrument as configured cannot detect contradiction signatures that exist. These states have opposite implications for the program and are not distinguishable from the present data. The protocol's staging prevented the expenditure that would have compounded the ambiguity: the full H2–H4 sweep was priced at roughly an order of magnitude above Stage 1 and was not run.

One specific mis-specification hypothesis is available and testable. Layer 24 of 32 sits at 75% of network depth. Layer-wise probing places syntactic structure in early-to-middle layers with semantic content higher [16, 17], and causal tracing localizes negation processing, the nearest well-studied relative of contradiction, to mid-stack layers [18], in GPT-2 at 33–50% of depth. If contradiction is represented as a relational feature in the same regime, the committed depth was above the band where the signature is most plausibly expressed. The commitment to a single blind depth was a fishing-prevention measure and served that purpose; it also concentrated all depth risk in one draw.

### 5.2 The necessity gate as the economic question

The motivating case for representation-level discord detection is partly economic, and its defensible form coincides with the scientific test. Downstream uncertainty methods carry a five-to-tenfold sampling overhead [35, 1]. The single-pass alternative is not hypothetical and is not topological: a linear probe on hidden states already recovers substantial semantic-uncertainty signal at near-zero marginal cost [35]. A topological read is also single-pass but computationally heavier than a linear one. It is therefore worth building only if it captures discord structure that the linear read does not, which is precisely the condition the necessity gate tests. A gate failure would establish that, for this phenomenon, the cheaper instrument suffices; a gate pass would establish reliability information unavailable at linear cost. The commercial and scientific questions are the same question, and neither is answered by the present data.

### 5.3 Pre-committed next test

A positive-control study is pre-committed here as the next test: its design, detection criteria, and outcome interpretations are fixed in this section in advance of any data, and its full protocol will be locked and released before the study executes. It replaces natural corpus items with minimal-contrastive engineered pairs, contradictions differing from their controls only in the contradicting element, matched on length, syntax, and vocabulary, across negation, antonym, numeric, and directional classes at two intensities, and it sweeps extraction depth across the mid-stack band implicated by [16, 17, 18] alongside the original layer 24. Its detection criterion is fixed in advance: the instrument is deemed functional only if the topological separation exceeds the position-only baselines (Section 3.5), appears across a contiguous band of depths rather than an isolated layer, and increases with contradiction intensity. The interpretation of each outcome is pre-committed: a clean detection licenses a return to natural-language testing with a calibrated instrument; a failure on maximal engineered contradictions localizes the Stage 1 null in the instrument and terminates the topological line of the program in favor of its other candidate instruments; an intermediate result is reported as a sensitivity-floor characterization. No outcome licenses reframing.

### 5.4 Relation to prior work

Two currents of prior work meet here. The first reads internal computation for semantic signal. Within it, the topological line has established that representation topology responds to adversarial perturbation [12, 21], hallucination [22], training dynamics [13], and prompt pathology [14], and that sentence and word embeddings carry non-trivial persistent structure [30, 31, 32, 33]. A survey documents the broader application to NLP [11]. The present protocol differs from this line in target (contradiction, with a dimensional-specificity claim), in standard (the necessity gate; no prior topological study of these phenomena tests against position-only baselines on identical data), and in inferential design (preregistered thresholds, matched coherent-edit controls, and enumerated falsification conditions). The nearest method is HalluZig [22], with which the secondary attention arm shares its zigzag-over-attention-graphs construction; that construction is due to [22] and its antecedents [15], and the present work claims only its subordination to the gate.

The second current intervenes on representations without retraining: inference-time steering along concept directions [23, 24, 25] and sparse-feature control [26, 27], with documented per-example brittleness [28] and capability trade-offs beyond a steering sweet spot [29]. No published method in either current holds a detected discord state as an internal object conditioning subsequent computation in the same pass; detection and intervention remain decoupled. The broader Murmur program targets that coupling. The present paper is confined to the detection prerequisite, and its operator-level architecture is out of scope here.

### 5.5 Limitations

The study tested one model family at one scale, one depth, one statistic, and pooled contradiction types; per-type analysis awaits a type classifier for the six-category taxonomy of [10]. Sample size (86 pairs, against 150 planned) gives Stage 1 power only against medium effects. The attention arm was not exercised at Stage 1. The instrument's sensitivity is unverified pending the positive control, and every interpretation above is conditional on that study's outcome. The term semantic discord organizes documented phenomena but is not itself validated as a natural kind; nothing in the experimental design depends on the term.

## Data and code availability

The analysis protocol (with amendment log), corpus manifests, and analysis outputs for Stage 1 are released with this paper at https://github.com/DylansAccord/dimensional-specificity. The positive-control protocol will be locked and released before that study runs. The operator-level specification of the Murmur architecture is covered by USPTO provisional filing 64/074,183.

## Acknowledgments and AI-assistance disclosure

This paper was prepared with substantial assistance from large language model systems: Anthropic's Claude (pipeline code, drafting, revision, and analysis support) and an OpenAI ChatGPT-based system (independent critical review, including the protocol revision that fixed the single confirmatory target and the corpus adequacy gate). No AI system qualifies for authorship. The research questions are the author's own; the experimental design and analytic decisions were directed and approved by the author, who is responsible for any remaining errors.

## References

[1] L. Kuhn, Y. Gal, S. Farquhar. Semantic uncertainty: linguistic invariances for uncertainty estimation in natural language generation. International Conference on Learning Representations, 2023. arXiv:2302.09664.

[2] S. Farquhar, J. Kossen, L. Kuhn, Y. Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.

[3] Y. Dong, M. Liu, M. Zhu, et al. Beyond semantic equivalence: logical graphs for LLM uncertainty quantification. arXiv:2607.16868, 2026.

[4] L. Da, T. Chen, L. Cheng, H. Wei. LLM uncertainty quantification through directional entailment graph and claim-level response augmentation. arXiv:2407.00994, 2024.

[5] X. Zhao, H. Peng, D. Su, X. Zeng, C. Liu, J. Liao, P. S. Yu. SeSE: black-box uncertainty quantification for large language models based on structural information theory. arXiv:2511.16275, 2025.

[6] G. Carlsson. Topology and data. Bulletin of the American Mathematical Society, 46(2):255–308, 2009.

[7] R. Ghrist. Barcodes: the persistent topology of data. Bulletin of the American Mathematical Society, 45(1):61–75, 2008.

[8] D. Cohen-Steiner, H. Edelsbrunner, J. Harer. Stability of persistence diagrams. Discrete & Computational Geometry, 37(1):103–120, 2007.

[9] U. Bauer. Ripser: efficient computation of Vietoris–Rips persistence barcodes. Journal of Applied and Computational Topology, 5(3):391–423, 2021.

[10] X. Wu, X. Niu, R. Rahman. Topological analysis of contradictions in text. Proceedings of the 45th International ACM SIGIR Conference, pp. 2478–2483, 2022.

[11] A. Uchendu, T. Le. Topological data analysis applications in natural language processing: a survey. SIGKDD Explorations, 2026. arXiv:2411.10298.

[12] A. Fay, I. García-Redondo, Q. Wang, H. Dubossarsky, A. Monod. The shape of adversarial influence: characterizing LLM latent spaces with persistent homology. International Conference on Learning Representations, 2026. arXiv:2505.20435.

[13] N. Malhotra, J. Ambadkar, A. Gupta, K. Kasivel, A. Schwarz, K. Ferry, A. Monod. Tracking representation dynamics in large language models with persistent homology. arXiv:2606.19542, 2026.

[14] G. Jiang, S. Tang, M. Imani, N. D. Bastian, T. Lan. TIPS: topological ill-posedness probing and steering in large language models. arXiv:2606.23590, 2026.

[15] Y. Gardinazzi, K. Viswanathan, G. Panerai, A. Ansuini, A. Cazzaniga, M. Biagetti. Persistent topological features in large language models. International Conference on Machine Learning, PMLR 267, 2025. arXiv:2410.11042.

[16] I. Tenney, D. Das, E. Pavlick. BERT rediscovers the classical NLP pipeline. Association for Computational Linguistics, pp. 4593–4601, 2019.

[17] B. van Aken, B. Winter, A. Löser, F. A. Gers. How does BERT answer questions? A layer-wise analysis of transformer representations. CIKM, 2019. arXiv:1909.04925.

[18] A. Al Mofael, L. M. Kuhn, G. Alkadi, K.-P. Yang. Interpreting negation in GPT-2: layer- and head-level causal analysis. IEEE Computing and Communication Workshop and Conference (CCWC), pp. 42–50, 2026. arXiv:2603.12423.

[19] A. Zomorodian, G. Carlsson. Computing persistent homology. Discrete & Computational Geometry, 33(2):249–274, 2005.

[20] M. Robinson, S. Dey, T. Chiang. Token embeddings violate the manifold hypothesis. Neural Information Processing Systems, 2025. arXiv:2504.01002.

[21] A. Tsai, S. Subramanian, C. Liu, K. Lopez, L. Zinn-Brooks, A. Schulz, A. Uchendu. The shape of vulnerability: how adversarial perturbations reshape the topology of language model latent spaces. ACL Student Research Workshop, pp. 290–306, 2026.

[22] S. N. Samaga, G. Gonzalez Arroyo, T. K. Dey. HalluZig: hallucination detection using zigzag persistence. EACL, pp. 3466–3482, 2026. arXiv:2601.01552.

[23] K. Li, O. Patel, F. Viégas, H. Pfister, M. Wattenberg. Inference-time intervention: eliciting truthful answers from a language model. Neural Information Processing Systems, 2023. arXiv:2306.03341.

[24] A. Zou, et al. Representation engineering: a top-down approach to AI transparency. arXiv:2310.01405, 2023.

[25] N. Rimsky, et al. Steering Llama 2 via contrastive activation addition. Association for Computational Linguistics, 2024. arXiv:2312.06681.

[26] A. Templeton, et al. Scaling monosemanticity: extracting interpretable features from Claude 3 Sonnet. Transformer Circuits, 2024.

[27] J. Ferrando, O. Obeso, S. Rajamanoharan, N. Nanda. Do I know this entity? Knowledge awareness and hallucinations in language models. International Conference on Learning Representations, 2025. arXiv:2411.14257.

[28] D. Tan, et al. Analyzing the generalization and reliability of steering vectors. Neural Information Processing Systems, 2024. arXiv:2407.12404.

[29] E. Durmus, et al. Evaluating feature steering: a case study in mitigating social biases. Anthropic, 2024.

[30] S. Das, S. A. Haque, M. I. Tanveer. Persistence homology of TEDtalk: do sentence embeddings have a topological shape? arXiv:2103.14131, 2021.

[31] S. Fitz, P. Romero, J. J. Schneider. Hidden holes: topological aspects of language models. arXiv:2406.05798, 2024.

[32] O. Draganov, S. Skiena. The shape of word embeddings: quantifying non-isometry with topological data analysis. Findings of EMNLP, 2024.

[33] E. Tulchinskii, et al. Intrinsic dimension estimation for robust detection of AI-generated texts. Neural Information Processing Systems, 2023.

[34] M. M. Bailey, M. I. Heiligman. Detecting narrative shifts through persistent structures: a topological analysis of media discourse. arXiv:2506.14836, 2025.

[35] J. Kossen, J. Han, M. Razzak, L. Schut, S. Malik, Y. Gal. Semantic entropy probes: robust and cheap hallucination detection in LLMs. arXiv:2406.15927, 2024.

[36] J. Hewitt, P. Liang. Designing and interpreting probes with control tasks. Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), pp. 2733–2743, 2019. arXiv:1909.03368.

[37] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, P. Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023.
