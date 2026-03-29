# p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding

**Authors:** Runyan Tan (Thoughtworks / NUS), Shuang Wu (Thoughtworks), Phillip Howard (Thoughtworks)
**Venue:** ICLR 2026 (Oral)
**arXiv:** 2509.23234v6

---

## What the paper does

Proposes p-less sampling, a truncation method requiring zero hyperparameters (beyond temperature). At each step, compute threshold L[P] = sum of squared token probabilities (related to Renyi entropy of order 2), admit only tokens with probability >= L[P], renormalize, sample. Claims to outperform min-p, top-p, top-k, epsilon-sampling, eta-sampling, mirostat.

Core theoretical idea is genuinely novel: connecting truncation thresholds to information-theoretic quantities (Renyi entropy, Friedman's Index of Coincidence) so the threshold adapts to the full distribution, not just the mode.

## Key claims

1. p-less "consistently outperforms existing sampling approaches" (abstract).
2. More robust to high temperatures than all baselines.
3. 22% faster inference than min-p (no sorting needed).
4. No hyperparameters to tune.

---

## Detailed Findings from Critical Analysis

### "Consistently outperforms" vs Table 1 reality

12 total comparisons (3 models × 4 datasets). On Llama3-70b (the largest model):

| Dataset | p-less | min-p | Winner |
|---------|--------|-------|--------|
| CSQA | 0.819 | **0.820** | min-p |
| GPQA | 0.387 | 0.377 | **p-less** |
| GSM8K | **0.932** | 0.930 | p-less |
| QASC | 0.894 | **0.899** | min-p |

**Min-p beats p-less on 2 of 4 datasets on Llama3-70b.** Margins are 0.001-0.005 AUC — within noise for single-seed evaluation.

### Single-seed evaluations on key models

From Appendix C.3: "The reported accuracies for Llama2-7b are averaged across generations produced by three different random seeds. For Mistral-7b and Llama3-70b, we provide the mean accuracy using one random seed due to computational constraints."

- **Llama-2-7b:** 3 seeds (the smallest, weakest model)
- **Mistral-7b:** 1 seed (no variance estimate possible)
- **Llama3-70b:** 1 seed (no variance estimate possible)

The strongest results are on the models with zero replication. Differences of 0.001 AUC on Llama3-70b are meaningless without variance estimates.

### p-less loses at T=1.0 on Llama3-70b (Table 5)

At the most practically relevant temperature on the largest model:

| Dataset | p-less | Best other | Winner |
|---------|--------|-----------|--------|
| CSQA | 81.7 | epsilon **82.6** | epsilon |
| GPQA | 38.2 | mirostat **41.1** | mirostat |
| GSM8K | **93.3** | min-p 93.0 | p-less |
| QASC | 89.0 | epsilon/top-p **90.6** | epsilon/top-p |

**p-less loses on 3 of 4 datasets at T=1.0 for Llama3-70b.** Its advantage is concentrated at high temperatures.

### AUC metric weights high temperatures disproportionately

AUC computed from only 5 temperature points: 0.5, 0.7, 1.0, 1.5, 2.0. With trapezoidal rule on these uneven spacings:
- T=0.5 to T=1.0: weight = 0.5 (1/3 of total range)
- T=1.0 to T=2.0: weight = 1.0 (2/3 of total range)

**2/3 of AUC weight comes from T >= 1.0**, where p-less has its biggest advantage and other methods degrade. This is never discussed or justified. No quadrature method is specified.

### Table 8: Tuned baselines approach p-less (Llama-2-7b only)

**Confirmed: Table 8 covers only Llama-2-7b.** Key comparisons:

| Dataset | p-less AUC | Best tuned baseline | Gap |
|---------|-----------|-------------------|-----|
| CSQA | 0.503 | min-p_0.05: 0.499 | 0.004 |
| GPQA | 0.248 | min-p_0.1: **0.249** | **min-p wins** |
| GSM8K | 0.267 | min-p_0.05: 0.264 | 0.003 |
| QASC | 0.537 | min-p_0.05: 0.521 | 0.016 |

On GPQA, tuned min-p actually beats p-less (0.249 vs 0.248). No tuned-baseline analysis for Mistral-7b or Llama3-70b.

### Confounded human evaluation

- Compares p-less at **T=2.0** vs default sampling (no truncation) at **T=1.0**
- 6 annotators: **3 are paper authors**
- Only 100 prompts, only Llama-2-7b
- Single generation per method per prompt
- Default sampling without truncation at T=1.0 is not a competitive baseline
- No inter-annotator agreement metric (Krippendorff's alpha, Cohen's kappa)
- Result: p-less wins 58.8% by majority vote

### Missing significance tests

- **No significance tests on any accuracy results** — the primary metric of the paper
- Significance tests DO exist for efficiency claims (Table 14: pairwise t-tests)
- The contrast is stark: they knew how to do significance tests and chose to apply them only where results were favorable

### Creative writing: p-less loses at T=1.0

Table 2: At T=1.0 on Llama-2-7b, epsilon-sampling wins creative writing with win rate **62.18** vs p-less **55.08**. Single generation per prompt, 100 prompts, no confidence intervals.

### No top-k baseline

Top-k — one of the most commonly used sampling methods — is not included in any comparison.

### Code repo lacks evaluation scripts

GitHub repo (https://github.com/ryttry/p-less) contains only: `p_less_samplers.py`, `p_less_examples.ipynb`, `LICENSE`, `README.md`. No evaluation scripts, no benchmark code, no sweep configs.

### Diversity/Pareto claim is thin

"Pareto dominance" claim (Figure 3) is based on 1 dataset (QASC), 1 model (Llama-2-7b), 5 temperature points. Table 10 shows min-p achieves higher diversity than p-less on Llama3-70b QASC at T=2.0 (78.7 vs 76.3).

---

## What p-less does well (fairness)

1. **Theoretical grounding:** Renyi entropy connection is elegant; proofs in Appendix B are rigorous.
2. **Bounded threshold guarantee:** Proposition 1 proves non-empty candidate set always.
3. **Efficiency analysis:** Thorough, with proper statistical testing (t-tests). O(|V|) complexity is a genuine advantage.
4. **Failure case analysis:** Appendix C.13 discusses two failure patterns — commendable transparency.
5. **The method itself is genuinely simple:** 3 lines of code, clear probabilistic interpretation.

---

## Mapping to our four standards

| Standard | Violation | Evidence |
|----------|-----------|----------|
| 1. Fair comparison | Baselines use default hyperparameters | Table 1 vs Table 8; tuned min-p matches p-less on GPQA |
| 2. Valid statistical inference | No significance tests on accuracy | Table 14 has tests for efficiency but not accuracy; 1-seed on key models |
| 3. Full transparency | Evaluation code not released | Only sampler code in GitHub repo |
| 4. Consistent reporting | AUC inflates high-temp advantage; "consistently outperforms" overclaimed | 2/3 AUC weight from T>=1.0; loses at T=1.0 on 3/4 datasets for Llama3-70b |

## Email to authors (2026-03-09, no response)

Rylan sent detailed methodological questions covering: (1) overclaimed "consistently outperforms" given mixed Llama3-70b results, (2) default baselines vs tuned baselines, (3) AUC inflated by high-temperature behavior, (4) confounded human eval, (5) missing significance tests, (6) missing citation of our work.
