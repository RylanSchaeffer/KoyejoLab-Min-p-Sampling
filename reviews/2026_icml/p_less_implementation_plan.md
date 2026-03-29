# P-less Sampling: Investigation & Implementation Plan

This document covers all planned investigations into p-less sampling (ICLR 2026 Oral, arXiv:2509.23234v6) for inclusion as a second case study in our manuscript.

---

## 1. The Algorithm

### P-less (standard)

At each decoding step, given token probability distribution P over vocabulary V:

```
threshold = sum(P(v)^2 for all v in V)    # = L[P], the Herfindahl index / collision probability
mask = P(v) >= threshold
P_truncated = P * mask
P_truncated = P_truncated / sum(P_truncated)   # renormalize
next_token = sample from P_truncated
```

In PyTorch:
```python
def p_less_truncate(probs):
    threshold = probs.square().sum(dim=-1, keepdim=True)
    mask = probs < threshold
    probs[mask] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs
```

This is the sum of squared probabilities, equivalent to exp(-H_2) where H_2 is the Renyi entropy of order 2. Also known as Friedman's Index of Coincidence or the Herfindahl-Hirschman Index.

### P-less_norm (relaxed variant)

```
threshold_norm = (|V| * sum(P(v)^2) - 1) / (|V| - 1)
```

This normalizes to account for vocabulary size, producing a lower threshold and admitting more tokens.

### Key properties

- Temperature IS a hyperparameter — despite the paper claiming "hyperparameter-free," temperature is swept from 0.5 to 2.0. P-less has 1 hyperparameter (temperature); other samplers have 2 (temperature + sampler-specific value).
- Threshold adapts to full distribution (not just the mode, unlike min-p).
- Non-empty guarantee: threshold <= max(P(v)), so at least the top token is always admitted.
- No sorting required: O(|V|) vs O(|V| log |V|) for min-p/top-p.

---

## 2. Investigations

### Investigation A: Best-of-N Evaluation (Standard 1)

**Question:** Does p-less's claimed advantage survive when baselines receive equal hyperparameter tuning?

**Approach:** Run p-less alongside our existing samplers (basic, top-p, top-k, min-p) on shared benchmarks and apply the Best-of-N protocol.

**Key design point:** P-less has fewer hyperparameter configurations than other samplers (only temperature, no sampler-specific value). At budget N:
- basic: can draw from 31 temperature configs
- p-less: can draw from 31 temperature configs (same as basic)
- top-p: can draw from 31 × 6 = 186 temperature × p-value configs
- top-k: can draw from 31 × 6 = 186 temperature × k-value configs
- min-p: can draw from 31 × 6 = 186 temperature × p-value configs

This is exactly the right test: p-less claims its advantage is that you don't need to tune. Best-of-N shows whether that advantage holds when other methods ARE tuned.

**Expected result:** If p-less's curve starts high (good default performance) but gets overtaken at moderate N, it demonstrates the same pattern as min-p: the advantage is an artifact of comparing a reasonable default against under-tuned baselines.

### Investigation B: Overclaimed "Consistently Outperforms" (Standard 4)

**Question:** Does the data in the paper actually support "consistently outperforms existing sampling approaches"?

**Findings (verified from Table 1):** On Llama3-70b:
- CSQA: min-p **0.820** vs p-less 0.819 → min-p wins
- QASC: min-p **0.899** vs p-less 0.894 → min-p wins
- GSM8K: p-less **0.932** vs min-p 0.930 → p-less wins
- GPQA: p-less_norm **0.391** vs p-less 0.387 → p-less itself loses to its own variant

**At T=1.0 on Llama3-70b (Table 5):** p-less loses on 3 of 4 datasets:
- CSQA: epsilon **82.6** vs p-less 81.7
- GPQA: mirostat **41.1** vs p-less 38.2
- GSM8K: p-less **93.3** (wins)
- QASC: epsilon/top-p **90.6** vs p-less 89.0

**Critical context:** Mistral-7b and Llama3-70b use only **1 random seed** (Appendix C.3). Llama-2-7b uses 3 seeds. The strongest results are on the models with zero replication. Margins of 0.001 AUC are meaningless with 1 seed.

**What to report:** Win/loss table across all comparisons, plus the 1-seed issue. "Consistently outperforms" is not supported by the paper's own data.

### Investigation C: AUC Metric Inflation (Standard 4)

**Question:** Does the AUC metric inflate p-less's practical advantage?

**Findings (verified):** AUC computed from only 5 temperature points: 0.5, 0.7, 1.0, 1.5, 2.0. With trapezoidal rule on these spacings:
- T=0.5 to T=1.0 contributes 1/3 of total range width
- T=1.0 to T=2.0 contributes 2/3 of total range width

**2/3 of AUC weight comes from T >= 1.0**, exactly where p-less has its advantage. No quadrature method specified. This is never discussed or justified.

Additionally: at T=1.0 on Llama-2-7b creative writing (Table 2), epsilon-sampling wins with **62.18** vs p-less **55.08**.

**What to report:** Side-by-side comparison of T=1.0 accuracy vs AUC. The metric choice inflates an advantage that doesn't exist at standard operating temperatures.

### Investigation D: Default Baselines vs Tuned Baselines (Standard 1)

**Question:** How much does p-less's advantage shrink when baselines are properly tuned?

**Findings (verified from Table 8, Llama-2-7b only):**

| Dataset | p-less AUC | Best tuned baseline | Gap |
|---------|-----------|-------------------|-----|
| CSQA | 0.503 | min-p_0.05: 0.499 | 0.004 |
| GPQA | 0.248 | min-p_0.1: **0.249** | min-p wins |
| GSM8K | 0.267 | min-p_0.05: 0.264 | 0.003 |
| QASC | 0.537 | min-p_0.05: 0.521 | 0.016 |

**Tuned min-p beats p-less on GPQA** (0.249 vs 0.248). Table 8 covers only Llama-2-7b — no tuned-baseline analysis for Mistral-7b or Llama3-70b.

**What to report:** For each dataset in Table 8, the best tuned-baseline AUC vs p-less AUC. This is the paper's own data showing that the advantage shrinks under tuning.

### Investigation E: Missing Significance Tests (Standard 2)

**Question:** Are the reported accuracy advantages statistically significant?

**Findings (verified):**
- **No significance tests on any accuracy results.** Table 1 (AUC), Table 5 (per-temperature), Table 2 (creative writing) — zero confidence intervals, standard errors, p-values, or significance tests.
- **Significance tests DO exist for efficiency** (Table 14: pairwise t-tests). p-less vs min-p p-value = 0.0011. p-less vs eta-sampling = 0.0486 (barely significant).
- The authors knew how to do significance tests and applied them only where results were favorable.
- Smallest margins in Table 1: CSQA Llama3-70b = 0.001 AUC difference. With 1 seed, this is indistinguishable from noise.

### Investigation F: Confounded Human Evaluation (Standard 3)

**Question:** Can p-less's human eval advantage be disentangled from the temperature difference?

**Findings (verified from Appendix A):**
- Compares p-less at **T=2.0** vs default sampling (no truncation) at **T=1.0**
- **3 of 6 annotators are paper authors**
- 100 prompts, Llama-2-7b only, single generation per prompt
- No comparison against min-p, top-p, or other truncation methods in human eval
- Default sampling without truncation at T=1.0 is not a competitive baseline
- No inter-annotator agreement metric (no Krippendorff's alpha, no Cohen's kappa)
- 23.7% unanimous agreement, 26.9% ties
- Author win rates: 57.6%, 54.3%, 57.1%. Non-author: 54.9%.
- Result: p-less wins 58.8% by majority vote

### Investigation G: No Top-k Baseline

Top-k — one of the most commonly used sampling methods — is not included in any comparison. The baselines are: top-p, min-p, epsilon, eta, mirostat. This is a notable omission.

### Investigation H: Diversity/Pareto Claim Is Thin

**Findings (verified):** "Pareto dominance" claim (Figure 3) based on:
- 1 dataset (QASC)
- 1 model (Llama-2-7b)
- 5 temperature points

Table 10 (Appendix) shows min-p achieves higher diversity than p-less on Llama3-70b QASC at T=2.0 (78.7 vs 76.3) with comparable accuracy. The Pareto claim does not generalize.

### Investigation I: Theoretical Strengths (Fairness)

**What is genuinely good about p-less — must acknowledge in manuscript:**
- Theoretical contribution: Renyi entropy connection is elegant; proofs in Appendix B are rigorous.
- Bounded threshold guarantee: Proposition 1 proves non-empty candidate set always.
- Efficiency: O(|V|) vs O(|V| log |V|) is real, with proper statistical testing.
- Failure case analysis in Appendix C.13 — commendable transparency.
- The method itself is simple: 3 lines of code, clear probabilistic interpretation.
- k-order generalization (Section B.5, Table 9) shows the method is part of a principled family.

---

## 3. Implementation: Adding P-less to Our Evaluation Infrastructure

### Option A: Patch vLLM's sampler (RECOMMENDED)

**Where to patch:** vLLM v0.7.3's sampling logic. The relevant file is likely `vllm/model_executor/layers/sampler.py` or the equivalent in v0.7.3's structure.

**What to add:**
1. Add `p_less: Optional[float]` to `SamplingParams`. Use a float flag where 0.0 = disabled, any positive value = enabled (could use 1.0 for standard p-less, 2.0 for p-less_norm, or just a bool).
2. In the sampler's `_apply_top_p_top_k` or equivalent function, add a p-less truncation step:
   - After temperature scaling and softmax (so we're operating on probabilities, not logits)
   - Compute threshold = probs.square().sum(dim=-1, keepdim=True)
   - Zero out tokens below threshold
   - Renormalize
3. This is ~15 lines of actual logic.

**How it flows through the stack:**
- `run_one_eval.py` already passes `--gen_kwargs sampler=value,temperature=X` to lm_eval
- lm_eval's vLLM wrapper passes gen_kwargs to `SamplingParams(**kwargs)`
- If `p_less` is a valid SamplingParams field, it flows through automatically
- In `run_one_eval.py`, handle `sampler == "p_less"` like `sampler == "basic"` (no sampler_value needed beyond the flag):
  ```python
  elif config["sampler"] == "p_less":
      gen_kwargs = f"p_less=1.0,temperature={config['temperature']},do_sample={do_sample}"
  ```

**Risk:** vLLM v0.7.3's V1 engine has a known bug (GitHub issue #12678) where custom `logits_processors` are silently ignored. A native SamplingParams field avoids this entirely.

**To find the exact file to patch:**
```bash
conda activate min_p_env
pip show vllm  # find install location
# Look at: vllm/sampling_params.py (add the parameter)
# Look at: vllm/model_executor/layers/sampler.py (add the truncation logic)
# grep for "min_p" to find where min-p is implemented — p-less goes in the same place
```

### Option B: Custom logits processor via lm_eval

**How:** Write a custom logits processor class and pass it via lm_eval's Python API (not CLI).

**Problem:** lm_eval's CLI (`--gen_kwargs`) only passes string key=value pairs. Cannot pass callable objects. Would need to fork lm_eval's vLLM wrapper to inject the processor, or write a custom model wrapper.

**Additional problem:** vLLM v0.7.3 V1 engine may silently ignore logits_processors (issue #12678).

**Verdict:** More fragile than Option A. Use only if patching vLLM is infeasible.

### Option C: Bypass vLLM/lm_eval entirely

**How:** Write a standalone eval script using HuggingFace Transformers with manual token-by-token generation (like the p-less authors did).

**Problem:** Different inference backend means results aren't directly comparable to our existing vLLM-based sweeps. Also much slower (no vLLM batching/optimization).

**Verdict:** Last resort. Use only if Options A and B fail.

---

## 4. Changes to Project Code

### `scripts/run_one_eval.py`

Add p-less handling in the gen_kwargs construction (around line 22-26):

```python
if config["sampler"] == "basic":
    gen_kwargs = f"temperature={config['temperature']},do_sample={do_sample}"
elif config["sampler"] == "p_less":
    gen_kwargs = f"p_less=1.0,temperature={config['temperature']},do_sample={do_sample}"
else:
    gen_kwargs = f"{config['sampler']}={config['sampler_value']},temperature={config['temperature']},do_sample={do_sample}"
```

### `src/globals.py`

Add to `SAMPLERS_NICE_NAMES_DICT`:
```python
"p_less": "P-less",
```

Add to `SAMPLERS_ORDER_LIST`:
```python
SAMPLERS_ORDER_LIST = [
    "Basic",
    "Top-p",
    "Top-k",
    "Min-p",
    "P-less",
]
```

Add to `MODELS_NICE_NAMES_DICT` (if adding new models for overlap with p-less paper):
```python
"meta-llama/Llama-2-7b-chat-hf": "Llama 2 7B Chat",
```

Add to `MODELS_TYPE_DICT`, `MODELS_ORDER_LIST` correspondingly.

### `src/analyze.py`

The `compute_diff_of_best_of_n_avg_scores_df` function (line 161-162) is hardcoded to compare "Min-p" vs everything else:
```python
min_p_rows = subset_of_at_most_N_per_sampler_df["Sampler"] == "Min-p"
```

For the p-less case study, we'd need a parallel version or a generalized version that takes the "target sampler" as a parameter:
```python
def compute_diff_of_best_of_n_avg_scores_df(
    runs_scores_df, target_sampler="Min-p", ...
):
    target_rows = subset_of_at_most_N_per_sampler_df["Sampler"] == target_sampler
```

---

## 5. Sweep Configuration

### What to sweep

P-less has no sampler-specific hyperparameter beyond temperature. Sweep config:

```yaml
program: scripts/run_one_eval.py
entity: rylan
project: min-p-evals
method: grid
parameters:
  model:
    values: ["vllm"]
  model_hf_path:
    values: ["mistralai/Mistral-7B-Instruct-v0.1"]
  num_fewshot:
    values: [8]
  sampler:
    values: ["p_less"]
  sampler_value:
    values: [0.0]  # placeholder, not used
  seed:
    values: [0, 1, 2]
  task:
    values: ["gsm8k_cot"]
  temperature:
    values: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
```

### Which models and benchmarks

**Minimum viable (for overlapping comparison with p-less paper):**
- Models: `mistralai/Mistral-7B-Instruct-v0.1` (shared with both our sweeps and p-less paper)
- Benchmarks: `gsm8k_cot`, `gpqa_main_generative_n_shot` (shared with both)
- Sweep size: 1 model × 31 temperatures × 3 seeds = 93 runs

**Stronger (all our existing models, overlapping benchmarks):**
- Models: All 18 models in our existing sweeps
- Benchmarks: `gsm8k_cot`, `gpqa_main_generative_n_shot`
- Sweep size: 18 models × 31 temperatures × 3 seeds = 1,674 runs

**Strongest (all models, all benchmarks):**
- Models: All 18 models + optionally Llama-2-7B-Chat for direct comparison
- Benchmarks: All 5 benchmark families (gsm8k_cot, gpqa, mmlu_pro, hendrycks_math, bbh_cot_fewshot)
- Sweep size: much larger, probably unnecessary for the rebuttal

**Recommendation:** Start with the minimum viable set (93 runs on Mistral-7B-Instruct). If results are clear, that's sufficient for the paper. Expand to more models if needed.

### Sweep YAML files to create

Place in `sweeps/nlp_benchmarks/gsm8k_cot/gsm8k_cot_p_less_part1.yaml` and `sweeps/nlp_benchmarks/gpqa/gpqa_p_less_part1.yaml`.

---

## 6. Execution Checklist

### Phase 1: Implementation
- [ ] Find vLLM install location in the min_p_env conda environment on the GPU cluster
- [ ] Grep for `min_p` in vLLM source to find where truncation samplers are implemented
- [ ] Add `p_less` to vLLM `SamplingParams` (sampling_params.py)
- [ ] Add p-less truncation logic to vLLM sampler (sampler.py, next to min_p)
- [ ] Update `run_one_eval.py` to handle `sampler == "p_less"`
- [ ] Update `src/globals.py` with p-less entries
- [ ] Test with a single run: `export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && python -u scripts/run_one_eval.py` with sampler="p_less" in EVAL_DEFAULT_CONFIG

### Phase 2: Sweeps
- [ ] Create sweep YAMLs for gsm8k_cot and gpqa with p_less
- [ ] Run sweeps on Mistral-7B-Instruct-v0.1 (minimum viable: 93 runs)
- [ ] Verify sweep results in W&B

### Phase 3: Analysis
- [ ] Generalize `src/analyze.py` `compute_diff_of_best_of_n_avg_scores_df` to accept target sampler parameter
- [ ] Run Best-of-N analysis combining p-less results with existing sweep data
- [ ] Generate plots showing p-less on the Best-of-N curves

### Phase 4: Paper Analysis (no compute needed)
- [ ] Tabulate all Table 1 comparisons: wins/losses/ties for p-less
- [ ] Extract Table 8 numbers showing tuned baselines approaching p-less
- [ ] Extract per-temperature results showing where p-less loses at T=1.0
- [ ] Document the confounded human eval (T=2.0 vs T=1.0, author annotators)
- [ ] Document missing significance tests for accuracy claims

### Phase 5: Writing
- [ ] Write 1-page case study section for the manuscript
- [ ] Write contamination chain paragraph for Discussion
- [ ] Add p-less to Related Work section

---

## 7. Estimated Compute

- Mistral-7B-Instruct on GSM8K CoT: ~20 min per run on A100 → 93 runs × 20 min = ~31 GPU-hours
- Mistral-7B-Instruct on GPQA: ~15 min per run on A100 → 93 runs × 15 min = ~23 GPU-hours
- **Total minimum viable: ~54 A100-hours** (trivial compared to the 6,000 hours for the main study)
- Expanding to all 18 models: ~18× = ~970 A100-hours
