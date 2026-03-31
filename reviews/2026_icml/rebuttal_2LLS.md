# Response to Reviewer 2LLS — ICML 2026 Submission 31762

---

## General Response (to all reviewers)

Three of four reviewers raise the same concern: the blueprint relies on a single case study. We address this directly, then describe cross-cutting revisions.

**Second case study: p-less sampling.** We have applied Best-of-N to p-less sampling (Tan et al., ICLR 2026 Oral, arXiv:2509.23234v6), which claims to "consistently outperform existing sampling approaches." We implemented p-less as a native sampler in vLLM v0.7.3 and are running sweeps across 28 models on GSM8K CoT, GSM8K CoT (Llama template), and GPQA — 5,952 runs on 3 seeds each. Best-of-N results will follow within 1-2 days.

P-less makes a genuinely novel theoretical contribution (connecting truncation thresholds to Renyi entropy) with real efficiency advantages (O(|V|) complexity). Our critique targets evaluation methodology, not the method's design.

Independent of the Best-of-N results, the p-less paper violates all four standards:
- **Standard 1 (Fair comparison):** Baselines use default hyperparameters. The paper's own Table 8 (Llama-2-7b only) shows tuned min-p matches or beats p-less on GPQA (0.249 vs 0.248). No tuned-baseline analysis exists for Mistral-7b or Llama3-70b. Top-k is omitted entirely.
- **Standard 2 (Valid inference):** No significance tests on any accuracy metric, despite including t-tests for efficiency claims (Table 14). Mistral-7B and Llama3-70B use only 1 random seed — making reported differences of 0.001 AUC uninterpretable.
- **Standard 3 (Transparency):** Human evaluation compares p-less at T=2.0 against default sampling at T=1.0, with 3 of 6 annotators being paper authors. No inter-annotator agreement reported. The public repository contains only the sampler implementation — no evaluation scripts, benchmark code, or sweep configurations.
- **Standard 4 (Consistent reporting):** "Consistently outperforms" is claimed despite min-p winning on 2/4 datasets for Llama3-70B (Table 1). At T=1.0 — the most practically relevant temperature — p-less loses on 3/4 accuracy datasets for Llama3-70b (Table 5: epsilon 82.6 vs p-less 81.4 on CSQA; mirostat 41.1 vs p-less 38.4 on GPQA; min-p 90.6 vs p-less 89.8 on QASC). "Excels in creative writing" is claimed, but at T=1.0 p-less ranks last among all 7 methods (Table 2). The AUC metric gives T=1.5 the single largest weight (33.3%) due to unevenly-spaced temperature points — the regime where p-less has its biggest advantage — never disclosed or justified. The title claims a "hyperparameter-free" approach, but temperature is swept from 0.5 to 2.0.

Different paper, different authors, different venue, same evaluation problems. The blueprint generalizes.

**Independent corroboration: Artificial Hivemind (NeurIPS 2025 Best Paper).** Jiang et al. (2025) tested min-p for output diversity (p=0.1, T=2.0) across 70+ models with 31,250 human annotations. 61% of response pairs exceeded 0.8 similarity — min-p does not meaningfully reduce mode collapse. This independently validates our finding that min-p's diversity claims are unsupported. Jiang et al. then concluded "decoding-time interventions are fundamentally insufficient" — an over-generalization resting on the unverified assumption that min-p adequately represents decoding-time methods. Flawed upstream evaluation propagates incorrect conclusions downstream.

**Cross-cutting revisions.** The revision will restructure the paper so the blueprint and Best-of-N protocol come first as general tools, with the two case studies as illustrations. Numerical discrepancies and author interactions move to the appendix. We will add a Related Work section covering hyperparameter search bias (Dodge et al., 2019; Bouthillier et al., 2021), benchmarking fairness (Henderson et al., 2018; Dehghani et al., 2021), statistical evaluation (Dror et al., 2018; Pineau et al., 2021), and fair comparison precedents (Melis et al., 2020). We will also add Algorithm 1 (Best-of-N pseudocode) and an operationalized checklist validated against both case studies.

---

## Response to Reviewer 2LLS

Thank you for the detailed and constructive review.

### Q1: Code availability

Yes. All code, sweep configurations, W&B sweep data, and analysis notebooks are publicly available at [repository URL, anonymized for review]. The p-less vLLM patch will be included. The revision will make the repository link prominent.

### Q2: Why equal-effort fairness?

The Best-of-N curve does not privilege one fairness philosophy — it reveals information relevant to several:

- **"Best achievable performance":** Read the right end of the curve (large N). If method A dominates at full budget, it is genuinely superior.
- **"Tuning is part of the method":** Read the left end (small N). Higher at N=1 means better defaults; faster rise means easier to tune.
- **"Equal effort":** Compare at any fixed N.

A single reported number (e.g., "min-p achieves X on GSM8K") collapses this into one point that can be gamed by choosing favorable hyperparameters. The curve makes the full picture visible; researchers can read off whichever comparison they prefer.

Limitation: the curve measures effort as configuration count, not wall-clock time. For methods with very different per-evaluation costs, a cost-adjusted comparison should supplement it. We will discuss this in the revision.

### Best-of-N: formalization and distinction from grid search

Grid search selects the best hyperparameters for one method; output is a single configuration. Best-of-N diagnoses whether a claimed advantage survives equalized tuning budgets; output is comparative performance-vs-budget curves. Same mechanics, different purpose — analogous to how pass@k (Chen et al., 2021) uses well-known subsampling but repurposes it as a diagnostic evaluation metric, or how Yang et al. (2025) repurposed pass@k to answer a new question about RL and reasoning. Our contribution is not the subsampling mechanism but its systematic application as a diagnostic protocol for detecting inflated claims.

The revision will add: (1) Algorithm 1 (pseudocode), (2) empirical analysis of how estimate variance decreases with N and the minimum N needed for stable rankings, and (3) explicit limitations — the protocol measures fairness in configuration count rather than compute cost, assumes the sweep covers the relevant region, and is most informative when methods have comparable numbers of hyperparameters.

### Best-of-N: related work

The revision will add a structured Related Work section. Positioning: Dodge et al. (2019) and Bouthillier et al. (2021) characterize the problem of unequal tuning but do not propose a reusable evaluation protocol. Henderson et al. (2018) demonstrate evaluation pitfalls in deep RL without a formalized comparison procedure. Melis et al. (2020) perform fair comparison ad hoc for their specific setting. Best-of-N systematizes these insights into a general-purpose diagnostic with a specific output format (comparative curves) applicable post-hoc to existing sweep data.

### Standards 2-4: operationalization

Agreed. The revision will include a concrete checklist mapping each standard to specific items, the failure mode each prevents, and where each was violated in both case studies:

- **Standard 2 (Valid inference) — Per-model significance tests with correction.** Prevents false discoveries from pooled/uncorrected tests. Violated in min-p (pooled t-tests across models) and p-less (no significance tests on any accuracy metric).
- **Standard 3 (Transparency) — Release all evaluation code and raw data.** Prevents selective reporting and irreproducibility. Partially violated in min-p (code released late); fully violated in p-less (no evaluation scripts, benchmark code, or sweep configurations released).
- **Standard 4 (Consistent reporting) — Win/loss tables across all comparisons.** Prevents cherry-picked metrics. Violated in min-p (omitted losing models) and p-less ("consistently outperforms" claimed despite min-p winning on 2/4 datasets for Llama3-70B).

Every item is validated against two independent papers, transforming the blueprint from commentary into a reusable tool.

### Tone, framing, and narrative restructuring

The revision will lead with the blueprint and Best-of-N protocol as general tools, with both case studies as illustrations. Numerical discrepancies move to appendix tables; author interactions to footnotes. The result: "a framework validated through two case studies" rather than "a critique that proposes a framework."

### Venue fit

The revised submission contains: (1) a formalized evaluation protocol with pseudocode, (2) 6,000+ A100-hours of new experiments across 28 models, 5 samplers, and 31 temperatures, (3) corrected statistical re-analyses overturning claims of a high-visibility ICLR 2025 Oral, (4) a second case study (p-less, ICLR 2026 Oral) demonstrating generality, and (5) an operationalized checklist validated against two independent papers. Evaluation methodology papers with this structure are regularly accepted at top ML venues: Henderson et al. (2018, AAAI), Dodge et al. (2019, EMNLP), Dehghani et al. (2021, NeurIPS).

### Computational cost

The 6,000 A100-hour figure is the cost of running sweeps, not the Best-of-N protocol itself. The protocol is post-hoc computation on existing sweep results (subsample N configurations, take the max, repeat). The sweeps represent the cost of fair comparison — cost that should be paid whenever a paper claims one method outperforms another. The protocol also degrades gracefully: even N=5-10 per method reveals whether a claimed advantage is robust or fragile.

### Reproducibility details

Seeds 0, 1, 2 across all runs. Hyperparameters swept on a fixed grid: 31 temperatures from 0.0 to 3.0 in 0.1 increments; sampler-specific values per sweep configuration files. All configurations enumerated, not randomly sampled. These details will be added to the revision.

### Presentation fixes

The revision will fix overflowing reference links, Figure 9 sizing, reference format inconsistencies, and typos.

---

## Summary of Planned Revisions

1. **Second case study (p-less):** Best-of-N experiments across 28 models demonstrating blueprint generality
2. **Algorithm 1:** Formal pseudocode, variance-vs-N analysis, minimum N for stable rankings, grid search distinction
3. **Operationalized checklist:** Each standard mapped to concrete items, failure modes, and violations in both case studies
4. **Related Work section:** Dodge 2019, Bouthillier 2021, Henderson 2018, Dehghani 2021, Dror 2018, Melis 2020
5. **Narrative restructuring:** Blueprint first, case studies as illustrations; min-p background; numerical discrepancies and author interactions moved to appendix
6. **Expanded Limitations:** Best-of-N limitations (configuration count vs. compute cost, search space coverage, heterogeneous hyperparameters)
7. **Presentation fixes:** URL formatting, Figure 9 sizing, reference format consistency, typos
8. **Wilcoxon robustness checks:** Non-parametric confirmation of all statistical conclusions
