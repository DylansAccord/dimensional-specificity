# Cloud Cost Budget

Target total: **$500** cloud spend, with $100 reserve for re-runs and overruns.
Expected actual: **$100-285**.

## Itemized estimates

### Sentence-embedding arm

| Component | Resource | Duration | Cost |
|---|---|---|---|
| Llama-3-8B base inference (4500 instances) | A10 24GB | 15-25 GPU-hrs | $11-19 |
| Llama-3-8B-Instruct inference (4500 instances) | A10 24GB | 15-25 GPU-hrs | $11-19 |
| Persistent homology computation | CPU instance (8 vCPU) | 100-200 CPU-hrs | $30-120 |
| Storage and data transfer | Block storage | persistent | $5-10 |
| **Subtotal** | | | **$57-168** |

### Attention-graph arm (pilot)

| Component | Resource | Duration | Cost |
|---|---|---|---|
| Llama-3-8B base attention extraction (900 instances) | A10 24GB | 10-20 GPU-hrs | $8-15 |
| Zigzag persistence computation | CPU instance (16 vCPU) | 50-100 CPU-hrs | $15-60 |
| **Subtotal** | | | **$23-75** |

### Bootstrap analysis and re-runs

| Component | Resource | Duration | Cost |
|---|---|---|---|
| 1000-iter bootstrap on both arms | CPU instance | 20-40 CPU-hrs | $10-25 |
| Anticipated re-runs (2x) | Mixed | — | $10-17 |
| **Subtotal** | | | **$20-42** |

### Total

| | Low | High |
|---|---|---|
| Sentence-embedding arm | $57 | $168 |
| Attention-graph arm | $23 | $75 |
| Bootstrap and re-runs | $20 | $42 |
| **Total expected** | **$100** | **$285** |
| Reserve | $100 | $100 |
| **Budget cap** | **$200** | **$385** |

## Provider pricing assumptions

Based on Lambda Labs and RunPod consumer cloud pricing as of June 2026:

- A10 24GB: $0.75/hr
- A100 40GB: $1.50/hr (alternative; used if Dionysus benefits from extra memory)
- CPU instance (8 vCPU, 32GB RAM): $0.30/hr
- CPU instance (16 vCPU, 64GB RAM): $0.60/hr
- Block storage: $0.10/GB-month

Hyperscaler pricing (AWS, GCP, Azure) is roughly 2-3x these rates. If using a
hyperscaler, double the budget estimates above; the total still fits within $500.

## Cost controls

1. **Dry run first.** Before launching the full experiment, run `configs/dry_run.yaml`
   end-to-end at small scale (~$5 spend). Catches pipeline bugs that would
   otherwise consume budget at full scale.

2. **Inference and PH compute on separate instances.** PH compute does not need
   GPU. Run inference on a GPU instance, save embeddings to block storage, shut
   down the GPU instance, then spin up a CPU instance for PH compute. Avoids
   paying for idle GPU time during the multi-hour PH phase.

3. **Spot/preemptible instances where possible.** Lambda Labs and RunPod offer
   spot pricing 30-50% below on-demand. Acceptable for batch jobs that can
   resume from checkpoint.

4. **Checkpointing.** All phases save intermediate state. If an instance is
   preempted or crashes, the pipeline can resume from the last completed item
   rather than restarting.
