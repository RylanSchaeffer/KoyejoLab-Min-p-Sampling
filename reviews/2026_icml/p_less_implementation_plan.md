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

## CRITICAL CONSTRAINT: Backwards Compatibility

**Everything MUST be backwards compatible.** All changes to vLLM, project scripts, and analysis code must preserve existing behavior for all current samplers (basic, top-p, top-k, min-p). Specifically:

- **vLLM `SamplingParams`:** `p_less` defaults to `0.0` (disabled). Existing configs that don't specify `p_less` are unaffected. When temperature is 0 (greedy), `p_less` is reset to `0.0` just like `min_p`.
- **vLLM sampler:** The `do_p_less` flag defaults to `False`. The p-less truncation function is only called when at least one request has `p_less > 0`. Zero performance overhead when not used.
- **`run_one_eval.py`:** The new `elif config["sampler"] == "p_less"` branch only triggers for p-less sweep configs. The existing `if/else` for basic and other samplers is unchanged.
- **`src/globals.py`:** Adding entries to dicts/lists is additive — existing entries remain, existing ordering preserved. New `"P-less"` is appended to the end of `SAMPLERS_ORDER_LIST`.
- **`src/analyze.py`:** The `compute_diff_of_best_of_n_avg_scores_df` function is generalized with a `target_sampler` parameter that defaults to `"Min-p"`, preserving the exact current behavior for all existing notebooks. The new parameter is only used explicitly in new p-less analysis notebooks.
- **Sweep YAMLs:** New YAML files added alongside existing ones. No existing YAMLs modified.
- **Notebooks:** All existing notebooks (`00-04`, `10-12`, `20`) continue to work without modification.

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

**Scope: Extensive — match all existing model/benchmark coverage.**

P-less has the same sweep footprint as "basic" (no sampler_value, just temperature × seeds = 93 runs per model). We mirror the existing part splits exactly:

| Part | Models |
|------|--------|
| Part 1 | Qwen2.5-{0.5B,1.5B,3B,7B}{,Instruct}, Mistral-7B-{v0.1,Instruct-v0.1} (10 models) |
| Part 2 | Llama-3.2-3B{,Instruct}, Llama-3.1-8B{,Instruct}, Gemma-2-{2b,2b-it,9b,9b-it} (8 models) |
| Part 3 | Qwen2.5-{14B,32B,72B}{,Instruct}, Gemma-2-{27b,27b-it}, Llama-3.1-70B{,Instruct} (10 models) |

Benchmarks and their part coverage (mirroring existing sweeps):

| Benchmark | num_fewshot | Parts | Runs per part | Total runs |
|-----------|-------------|-------|---------------|------------|
| gsm8k_cot | 8 | 1,2,3 | 930, 744, 930 | **2,604** |
| gsm8k_cot_llama | 8 | 1,2 | 930, 744 | **1,674** |
| gpqa | 5 | 1,2 | 930, 744 | **1,674** |

**Grand total: 7 YAML files, 5,952 runs.**

### Sweep YAML files to create

```
sweeps/nlp_benchmarks/gsm8k_cot/gsm8k_cot_p_less_part1.yaml
sweeps/nlp_benchmarks/gsm8k_cot/gsm8k_cot_p_less_part2.yaml
sweeps/nlp_benchmarks/gsm8k_cot/gsm8k_cot_p_less_part3.yaml
sweeps/nlp_benchmarks/gsm8k_cot_llama/gsm8k_cot_llama_p_less_part1.yaml
sweeps/nlp_benchmarks/gsm8k_cot_llama/gsm8k_cot_llama_p_less_part2.yaml
sweeps/nlp_benchmarks/gpqa/gpqa_p_less_part1.yaml
sweeps/nlp_benchmarks/gpqa/gpqa_p_less_part2.yaml
```

Each follows the exact same template as the corresponding `_basic_partN.yaml` but with `sampler: ["p_less"]`.

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

---

## 8. Detailed Implementation Spec (from code reading)

### Environment & vLLM Location

- Conda env: `min_p_env` (Python 3.11)
- vLLM version: **0.7.3**
- vLLM install: `/lfs/skampere2/0/rschaef/miniconda3/envs/min_p_env/lib/python3.11/site-packages/vllm/`
- lm_eval version: 0.4.7
- Default engine: **V0** (`VLLM_USE_V1=0` by default)
- `SamplingParams` is a **`msgspec.Struct`** (not a dataclass) — fields are positional; adding a new field requires care with ordering and `from_optional()`

### gen_kwargs Flow

1. `run_one_eval.py` constructs gen_kwargs string: `"p_less=1.0,temperature=0.5,do_sample=True"`
2. lm_eval parses the string into a dict of key=value pairs
3. `lm_eval/models/vllm_causallms.py:modify_gen_kwargs()` pops `do_sample`, adjusts temperature
4. Remaining kwargs passed to `SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)`
5. Since `SamplingParams` is a msgspec.Struct, **kwargs are matched by name — so `p_less=1.0` will set the `p_less` field if it exists

### Files to Modify (exact paths and line numbers)

#### File 1: `vllm/sampling_params.py` — Add p_less field

**Location:** `/lfs/skampere2/0/rschaef/miniconda3/envs/min_p_env/lib/python3.11/site-packages/vllm/sampling_params.py`

Changes:
- **Line ~180:** Add `p_less: float = 0.0` after `min_p: float = 0.0`
- **Line ~223 (`from_optional`):** Add `p_less: float = 0.0` parameter, pass to constructor
- **Line ~265:** Add `p_less=p_less` in the `SamplingParams(...)` constructor call
- **Line ~347 (`__post_init__`):** When temperature < eps (greedy), add `self.p_less = 0.0` alongside `self.min_p = 0.0`
- **Line ~378 (`_verify_args`):** Add validation: `if not 0.0 <= self.p_less <= 1.0: raise ValueError(...)`
- **Line ~479 (`__repr__`):** Add `f"p_less={self.p_less}, "` after the min_p line
- **Docstring (~124):** Add description of the p_less parameter

#### File 2: `vllm/model_executor/layers/sampler.py` — V0 engine truncation

**Location:** `/lfs/skampere2/0/rschaef/miniconda3/envs/min_p_env/lib/python3.11/site-packages/vllm/model_executor/layers/sampler.py`

Changes:
- **Line ~207-214 (`_init_sampling_tensors`):** Unpack `do_p_less` from `SamplingTensors.from_sampling_metadata()`, store as `self._do_p_less`
- **Line ~256:** Add `do_p_less = self._do_p_less`
- **Line ~278 (after min_p block):** Add:
  ```python
  if do_p_less:
      logits = _apply_p_less(logits, sampling_tensors.p_less_vals)
  ```
- **After `_apply_min_p` function (line ~431):** Add new function:
  ```python
  def _apply_p_less(
      logits: torch.Tensor,
      p_less_vals: torch.Tensor,
  ) -> torch.Tensor:
      """Apply p-less truncation sampling (Hewitt et al., ICLR 2026).
      Threshold = sum of squared probabilities (Herfindahl index)."""
      probs = torch.softmax(logits, dim=-1)
      threshold = probs.square().sum(dim=-1, keepdim=True)
      tokens_to_remove = probs < threshold
      logits = logits.masked_fill_(tokens_to_remove, -float("inf"))
      return logits
  ```
  Note: The `p_less_vals` tensor is not used in the threshold computation (p-less is parameter-free). It only serves as a flag. The actual filtering is `do_p_less` gating above. But we carry the tensor to follow the same pattern as min_p for consistency. We could alternatively just use the bool flag — either works.

  **Simplification:** Since p-less has no tunable value (unlike min_p which scales by the parameter), we don't actually need the tensor values in the computation. The `do_p_less` bool is sufficient. But carrying the tensor through SamplingTensors keeps the pattern uniform.

#### File 3: `vllm/model_executor/sampling_metadata.py` — V0 tensors

**Location:** `/lfs/skampere2/0/rschaef/miniconda3/envs/min_p_env/lib/python3.11/site-packages/vllm/model_executor/sampling_metadata.py`

Changes to `SamplingTensors` dataclass:
- **Line ~378:** Add `p_less_vals: torch.Tensor` field
- **Line ~398:** Add `p_less_vals: List[float] = []` in `from_sampling_metadata`
- **Line ~404:** Add `do_p_less = False`
- **Line ~415:** Add `p_less_val = sampling_params.p_less`
- **Line ~428:** Add `if not do_p_less and p_less_val > _SAMPLING_EPS: do_p_less = True`
- **Line ~445 (prefill):** Add `p_less_vals += [p_less_val] * prefill_len`
- **Line ~456 (sample):** Add `p_less_vals += [p_less_val] * sample_lens`
- **Line ~484 (`from_lists` call):** Add `p_less_vals` argument
- **Line ~494 (return):** Change to `return (sampling_tensors, do_penalties, do_top_p_top_k, do_min_p, do_p_less)`
- **Line ~502 (`from_lists` signature):** Add `p_less_vals: List[float]` parameter
- **After min_ps_t tensor creation (~550-555):** Add:
  ```python
  p_less_vals_t = torch.tensor(p_less_vals, device="cpu", dtype=dtype, pin_memory=pin_memory)
  ```
- **Line ~583 (constructor return):** Add `p_less_vals=p_less_vals_t.to(device=device, non_blocking=True)`

**IMPORTANT:** The return tuple from `from_sampling_metadata` changes from 4 to 5 elements. The caller in `sampler.py` must be updated to unpack all 5.

#### File 4: `vllm/v1/sample/metadata.py` — V1 metadata

**Location:** `/lfs/skampere2/0/rschaef/miniconda3/envs/min_p_env/lib/python3.11/site-packages/vllm/v1/sample/metadata.py`

Changes:
- **Line ~21:** Add `p_less: Optional[torch.Tensor]` after `min_p`

#### File 5: `vllm/v1/sample/sampler.py` — V1 engine truncation

**Location:** `/lfs/skampere2/0/rschaef/miniconda3/envs/min_p_env/lib/python3.11/site-packages/vllm/v1/sample/sampler.py`

Changes:
- **Line ~108 (after min_p block):** Add:
  ```python
  # Apply p_less.
  if sampling_metadata.p_less is not None:
      logits = self.apply_p_less(logits, sampling_metadata.p_less)
  ```
- **After `apply_min_p` method (~line 214):** Add:
  ```python
  def apply_p_less(
      self,
      logits: torch.Tensor,
      p_less: torch.Tensor,
  ) -> torch.Tensor:
      """Apply p-less truncation (Hewitt et al., ICLR 2026)."""
      probability_values = torch.nn.functional.softmax(logits, dim=-1)
      threshold = probability_values.square().sum(dim=-1, keepdim=True)
      valid_token_mask = probability_values >= threshold
      logits[~valid_token_mask] = -float('inf')
      return logits
  ```

#### File 6: `vllm/v1/worker/gpu_input_batch.py` — V1 batch management

**Location:** `/lfs/skampere2/0/rschaef/miniconda3/envs/min_p_env/lib/python3.11/site-packages/vllm/v1/worker/gpu_input_batch.py`

Mirror all `min_p` patterns for `p_less`:
- **~Line 126-134:** Add `p_less`, `p_less_cpu_tensor`, `p_less_cpu`, `p_less_reqs` (same pattern as min_p)
- **~Line 257-261:** Add `self.p_less_cpu[req_index] = sampling_params.p_less` and req tracking
- **~Line 314:** Add `self.p_less_reqs.discard(req_id)`
- **~Line 387:** Add `self.p_less_cpu[empty_index] = self.p_less_cpu[last_req_index]`
- **~Line 418-419:** Add `if not self.no_p_less: copy_slice(self.p_less_cpu_tensor, self.p_less, num_reqs)`
- **~Line 445:** Add `p_less=None if self.no_p_less else self.p_less[:num_reqs]` to SamplingMetadata constructor
- **~Line 531-532:** Add `no_p_less` property: `return len(self.p_less_reqs) == 0`

#### File 7: `scripts/run_one_eval.py` — Project eval script

**Location:** `/lfs/skampere2/0/rschaef/KoyejoLab-Min-p-Sampling/scripts/run_one_eval.py`

Changes (line ~23-26):
```python
if config["sampler"] == "basic":
    gen_kwargs = f"temperature={config['temperature']},do_sample={do_sample}"
elif config["sampler"] == "p_less":
    gen_kwargs = f"p_less=1.0,temperature={config['temperature']},do_sample={do_sample}"
else:
    gen_kwargs = f"{config['sampler']}={config['sampler_value']},temperature={config['temperature']},do_sample={do_sample}"
```

**Key design:** Like `basic`, `p_less` does not use `config['sampler_value']`. The `p_less=1.0` is a flag value (any positive float triggers it). The sweep YAML uses `sampler_value: [0.0]` as a placeholder.

#### File 8: `src/globals.py` — Project globals

**Location:** `/lfs/skampere2/0/rschaef/KoyejoLab-Min-p-Sampling/src/globals.py`

Changes:
- **Line ~99 (`SAMPLERS_NICE_NAMES_DICT`):** Add `"p_less": "P-less",`
- **Line ~106 (`SAMPLERS_ORDER_LIST`):** Add `"P-less",` at the end (after "Min-p")

#### File 9: `src/analyze.py` — Analysis functions (deferred to Phase 3)

**Location:** `/lfs/skampere2/0/rschaef/KoyejoLab-Min-p-Sampling/src/analyze.py`

Changes to `compute_diff_of_best_of_n_avg_scores_df` (line ~106):
- Add `target_sampler: str = "Min-p"` parameter with default preserving current behavior
- Replace hardcoded `"Min-p"` references with `target_sampler` parameter
- Rename column from `"Best Min-p Exact Match - Best Other Exact Match"` to a generic name, or parameterize it

Changes to `compute_samplers_pairwise_scores_differences_df` (line ~203):
- Add `target_sampler: str = "Min-p"` parameter
- Replace hardcoded `sampler1 = "Min-p"` with `sampler1 = target_sampler`

**These changes are backwards-compatible** because the default parameter value preserves the current behavior.

### Sweep YAML Files to Create (7 files)

All p-less YAMLs mirror the corresponding `_basic_partN.yaml` exactly (same models, same num_fewshot, same temperatures, same seeds) but with `sampler: ["p_less"]` and `sampler_value: [0.0]`.

#### gsm8k_cot (3 files, num_fewshot=8)

**`sweeps/nlp_benchmarks/gsm8k_cot/gsm8k_cot_p_less_part1.yaml`** — mirrors `gsm8k_cot_basic_part1.yaml`
- Models: Qwen2.5-{0.5B,0.5B-Instruct,1.5B,1.5B-Instruct,3B,3B-Instruct,7B,7B-Instruct}, Mistral-7B-{v0.1,Instruct-v0.1}
- 10 models × 31 temps × 3 seeds = **930 runs**

**`sweeps/nlp_benchmarks/gsm8k_cot/gsm8k_cot_p_less_part2.yaml`** — mirrors `gsm8k_cot_basic_part2.yaml`
- Models: Llama-3.2-3B{,-Instruct}, Llama-3.1-8B{,-Instruct}, Gemma-2-{2b,2b-it,9b,9b-it}
- 8 models × 31 temps × 3 seeds = **744 runs**

**`sweeps/nlp_benchmarks/gsm8k_cot/gsm8k_cot_p_less_part3.yaml`** — mirrors `gsm8k_cot_basic_part3.yaml`
- Models: Qwen2.5-{14B,14B-Instruct,32B,32B-Instruct,72B,72B-Instruct}, Gemma-2-{27b,27b-it}, Llama-3.1-70B{,-Instruct}
- 10 models × 31 temps × 3 seeds = **930 runs**

#### gsm8k_cot_llama (2 files, num_fewshot=8)

**`sweeps/nlp_benchmarks/gsm8k_cot_llama/gsm8k_cot_llama_p_less_part1.yaml`** — mirrors `gsm8k_cot_llama_basic_part1.yaml`
- Same 10 models as gsm8k_cot part 1 = **930 runs**

**`sweeps/nlp_benchmarks/gsm8k_cot_llama/gsm8k_cot_llama_p_less_part2.yaml`** — mirrors `gsm8k_cot_llama_basic_part2.yaml`
- Same 8 models as gsm8k_cot part 2 = **744 runs**

#### gpqa (2 files, num_fewshot=5)

**`sweeps/nlp_benchmarks/gpqa/gpqa_p_less_part1.yaml`** — mirrors `gpqa_basic_part1.yaml`
- Same 10 models as gsm8k_cot part 1 = **930 runs**

**`sweeps/nlp_benchmarks/gpqa/gpqa_p_less_part2.yaml`** — mirrors `gpqa_basic_part2.yaml`
- Same 8 models as gsm8k_cot part 2 = **744 runs**

**Grand total: 5,952 runs across 7 YAML files.**

### Implementation Order

1. **vLLM SamplingParams** (File 1) — must come first, everything depends on the parameter existing
2. **vLLM V0 sampling_metadata** (File 3) — tensors for V0 engine
3. **vLLM V0 sampler** (File 2) — truncation logic for V0 engine
4. **vLLM V1 metadata** (File 4) — field for V1 engine
5. **vLLM V1 sampler** (File 5) — truncation logic for V1 engine
6. **vLLM V1 gpu_input_batch** (File 6) — batch management for V1 engine
7. **run_one_eval.py** (File 7) — project-level eval handling
8. **globals.py** (File 8) — display names
9. **Sweep YAMLs** — create all 7 YAML files
10. **analyze.py** (File 9) — deferred to Phase 3, not needed for running sweeps

### Testing Plan

After implementation, verify with a single dry-run:
```bash
export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && python -u scripts/run_one_eval.py
```
With `EVAL_DEFAULT_CONFIG` temporarily set to `sampler: "p_less"`, `temperature: 1.0`.

Check that:
1. No errors in vLLM sampler construction
2. `SamplingParams(p_less=1.0, temperature=1.0)` is accepted
3. Scores are logged to W&B
4. Running with `sampler: "basic"` still works (backwards compatibility)
