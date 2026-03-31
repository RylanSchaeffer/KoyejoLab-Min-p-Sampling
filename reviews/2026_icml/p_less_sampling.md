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
| CSQA | 81.4 | epsilon **82.6** | epsilon |
| GPQA | 38.4 | mirostat **41.1** | mirostat |
| GSM8K | **93.3** | min-p 92.4 | p-less |
| QASC | 89.8 | min-p **90.6** | min-p |

**p-less loses on 3 of 4 datasets at T=1.0 for Llama3-70b.** Its advantage is concentrated at high temperatures.

### AUC metric weights high temperatures disproportionately

AUC computed from only 5 temperature points: 0.5, 0.7, 1.0, 1.5, 2.0. Trapezoidal integration on these unevenly-spaced points gives each temperature the following effective weight:

| Temperature | Effective weight | Notes |
|-------------|-----------------|-------|
| T=0.5 | 6.7% | |
| T=0.7 | 16.7% | |
| T=1.0 | 26.7% | Most practical temperature |
| T=1.5 | **33.3%** | Largest weight; p-less's biggest advantage |
| T=2.0 | 16.7% | |

**T=1.5 receives the single largest weight** — the exact temperature regime where p-less gains its biggest advantage as other methods degrade. Half the total AUC weight (50%) comes from T > 1.0. Only 23.3% comes from T < 1.0 (where methods are most similar).

The paper never discloses these weights, never specifies a quadrature method, and never justifies why high-temperature performance should receive disproportionate weight. No search for "weight," "trapezoidal," "quadrature," or "spacing" yields any results in the paper. The AUC metric choice and temperature point selection are presented without discussion as "fair comparison."

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

- Compares p-less at **T=2.0** vs default sampling (no truncation) at **T=1.0** — different temperatures, fundamentally unfair
- 6 annotators: **3 are paper authors** (50%)
- Only 100 prompts, only Llama-2-7b
- Single generation per method per prompt
- Default sampling without truncation at T=1.0 is not a competitive baseline — compare against epsilon or top-p at T=1.0 (which beat p-less per Table 2)
- No inter-annotator agreement metric (Krippendorff's alpha, Cohen's kappa)
- Result: p-less wins 58.8% by majority vote

**Annotation procedure has undisclosed details.** Appendix A states 6 annotators produced "4 labels for each story pair." With 6 annotators but 4 labels per pair, only a subset annotated each pair — the selection procedure is not described. 26.9% of pairs received a "tie" (presumably 2-2 splits among 4 annotators). The paper says "for the remaining stories, we use the majority vote" but never explains how the 26.9% ties are resolved — majority vote is undefined for 2-2 splits. The 58.8% overall win rate includes these undisclosed tie resolutions.

### Missing significance tests

- **No significance tests on any accuracy results** — the primary metric of the paper
- Significance tests DO exist for efficiency claims (Table 14: pairwise t-tests)
- The contrast is stark: they knew how to do significance tests and chose to apply them only where results were favorable

### Creative writing: p-less is dead last at T=1.0, but paper claims it "excels"

Table 2, T=1.0, Llama-2-7b — length-controlled win rates:

| Rank | Method | Win rate |
|------|--------|----------|
| 1 | epsilon | 62.18 |
| 2 | top-p | 62.07 |
| 3 | eta | 58.76 |
| 4 | p-lessnorm | 58.74 |
| 5 | min-p | 57.48 |
| 6 | mirostat | 56.94 |
| **7** | **p-less** | **55.08** |

**p-less ranks 7th out of 7 methods at T=1.0** — the standard operating temperature for creative writing. Its "advantage" appears only at T≥1.5 where epsilon, eta, and top-p collapse to near-zero (literally 0.00 at T=2.0) because they produce degenerate text. P-less "winning" at T=2.0 means only that aggressive truncation prevents gibberish at extreme temperatures — not that the method produces superior creative writing.

Yet Section 4.3 states: "p-less remains relatively stable and is superior to all other methods at temperatures > 1.0. **This demonstrates how p-less excels in the domain of creative writing.**" The paper's own Table 2 directly contradicts this claim at the most practical temperature. Single generation per prompt, 100 prompts, no confidence intervals.

### No top-k baseline

Top-k — one of the most commonly used sampling methods — is not included in any comparison.

### Code repo lacks evaluation scripts

GitHub repo (https://github.com/ryttry/p-less) contains only: `p_less_samplers.py`, `p_less_examples.ipynb`, `LICENSE`, `README.md`. No evaluation scripts, no benchmark code, no sweep configs.

### Diversity/Pareto claim is thin

"Pareto dominance" claim (Figure 3) is based on 1 dataset (QASC), 1 model (Llama-2-7b), 5 temperature points. Table 10 shows min-p achieves higher diversity than p-less on Llama3-70b QASC at T=2.0 (81.9 vs 77.8).

### False claim that baselines "do not consider the output token distribution"

Section 3.4 states: "p-less contrasts with other methods that do not consider the output token distribution (e.g. top-k, top-p, ε-sampling, min-p)."

This is factually wrong for top-p and min-p:
- **Top-p** explicitly operates on the CDF of the output distribution — it accumulates probabilities from the most likely tokens until reaching threshold p. The number of admitted tokens varies with the distribution at every step. Top-p absolutely "considers the output token distribution."
- **Min-p** computes its threshold as `min_p × p_max`, where p_max is the maximum probability in the current output distribution. This is a first-order statistic of the distribution.

The accurate distinction: p-less uses a second-order summary statistic (Σp_i², the Herfindahl index) while min-p uses a first-order one (max p_i). That is a difference in degree, not the categorical difference the paper claims. Mischaracterizing baselines to inflate novelty is a methodological problem in the presentation, not a nitpick — it overstates the conceptual gap between p-less and existing work.

### Reproducibility Statement contradicts actual code release

Page 11: "We will make our source code publicly available upon publication in order to facilitate future efforts to reproduce our main experimental results." And: "the documentation in this manuscript contains all details necessary to fully reproduce our results."

The paper is published (ICLR 2026). The GitHub repo (https://github.com/ryttry/p-less) contains only: `p_less_samplers.py`, `p_less_examples.ipynb`, `LICENSE`, `README.md`. No evaluation scripts, no benchmark harness, no sweep configurations, no analysis code. The reproducibility assurance is factually unfulfilled.

### Section 5.4 case study is a single cherry-picked anecdote

The "Robustness Under High Entropy" case study (Section 5.4) presents ONE GSM8K example at T=2.0 where min-p produces an incorrect answer and p-less produces the correct one. This single anecdote is presented with entropy plots as if it constitutes systematic evidence. T=2.0 is wildly impractical for math reasoning — no practitioner would use this setting. The section title frames an extreme operating condition as a normal use case.

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

## Investigations for our sweeps (28 models, 3 seeds, 5,952 runs)

1. **Best-of-N curves:** Does p-less survive equalized tuning budgets? This is the core question.
2. **Per-temperature comparison at T=1.0:** Their Table 5 shows p-less loses 3/4 datasets at T=1.0 on Llama3-70b. Does this pattern hold across 28 models?
3. **"Consistently outperforms" across 28 models:** The abstract claims "consistently outperforms existing sampling approaches." Their evidence: 3 models, with 1-seed on 2 of them. We have 28 models with 3 seeds each. Count win/loss across all (model, dataset, temperature) triples.
4. **Is p-less equivalent to greedy at low temperatures?** By construction, low T → peaked distribution → high threshold → few tokens admitted. Check if p-less at T≤0.5 produces identical results to temperature-only sampling (no truncation).
5. **Recompute AUC with equal temperature spacing:** Our sweeps use T=0.0 to T=3.0 in 0.1 increments. Compute AUC with uniform spacing to eliminate the metric gaming.

## Email to authors (2026-03-09, no response)

Rylan sent detailed methodological questions covering: (1) overclaimed "consistently outperforms" given mixed Llama3-70b results, (2) default baselines vs tuned baselines, (3) AUC inflated by high-temperature behavior, (4) confounded human eval, (5) missing significance tests, (6) missing citation of our work.
