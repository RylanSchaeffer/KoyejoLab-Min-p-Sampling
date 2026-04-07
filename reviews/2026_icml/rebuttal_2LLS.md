# Response to Reviewer 2LLS

## General Response

Three of four reviewers note that the blueprint relies on a single case study. We have added a second: p-less sampling (Tan et al., ICLR 2026 Oral), which claims to "consistently outperform existing sampling approaches." We implemented p-less in vLLM v0.7.3 and ran Best-of-N sweeps across 18 models on GSM8K and GPQA (5,022 runs, 3 seeds each). Under equal tuning budget, p-less's best configuration loses to the best other sampler on 40/45 model-benchmark pairs and wins on only 7/45.

The p-less paper violates all four standards. Baselines use default hyperparameters only; their own Table 8 shows tuned min-p matches p-less on GPQA. Accuracy comparisons omit significance tests despite including them for efficiency. Human evaluation compares T=2.0 vs T=1.0, with 3 of 6 annotators being paper authors. And "consistently outperforms" is claimed when min-p wins 2/4 datasets on Llama3-70B and p-less loses 3/4 at T=1.0. Different paper, different authors, different venue, same problems.

Separately, Jiang et al. (NeurIPS 2025 Best Paper) independently tested min-p for diversity across 70+ models and found 61% of response pairs exceeded 0.8 similarity. This corroborates our finding that min-p's diversity claims are unsupported.

The revision restructures the paper so the blueprint comes first, with case studies as illustrations. We are adding a Related Work section, Algorithm 1 (Best-of-N pseudocode), and a checklist validated against both case studies.

---

## Responses to Specific Points

**Q1: Code availability.** Yes. All code, sweep configs, W&B data, and analysis notebooks will be made publicly available upon acceptance. The p-less vLLM patch will be included.

**Q2: Why equal-effort fairness?** The Best-of-N curve does not commit to one fairness philosophy. It shows performance as a function of tuning budget, which is informative under several views. The right end of the curve (large N) shows best achievable performance. The left end (small N) shows how easy a method is to tune. Any fixed N gives an equal-effort comparison. A single reported number collapses this into one cherry-pickable point. The curve makes the full picture visible. We now note as a limitation that effort is measured by configuration count, not compute.

**Best-of-N vs. grid search.** Grid search is an optimization procedure: pick the best hyperparameters for one method. Best-of-N is an evaluation procedure: test whether a claimed advantage survives when all methods receive equal tuning budget. It produces comparative performance-vs-budget curves across methods. The revision adds Algorithm 1 (pseudocode) and analysis of how variance decreases with N.

**Related work.** Dodge et al. (2019) and Bouthillier et al. (2021) characterize unequal tuning but propose no reusable protocol. Henderson et al. (2018) show RL evaluation pitfalls without a formalized comparison method. Melis et al. (2020) do fair comparison on an ad hoc basis. Best-of-N systematizes these insights into a general diagnostic that can be applied post-hoc to existing sweep data. The revision adds a structured Related Work section covering these and other evaluation methodology papers.

**Operationalizing Standards 2-4.** The revision adds a checklist mapping each standard to concrete items and failure modes, validated against both case studies. Standard 2: per-model significance tests with multiple comparison correction (min-p pooled across models; p-less omitted tests entirely). Standard 3: release all evaluation code and data (p-less released neither). Standard 4: report win/loss tables across all comparisons (both papers overclaimed). Scaling enforcement through automated tools is work in progress.

**Tone and framing.** We agree. The blueprint and protocol will now come first as general tools. Numerical discrepancies move to the appendix. Author interactions move to footnotes.

**Venue fit.** The revised paper contributes a formalized protocol with pseudocode, 6,000+ A100-hours of experiments across 28 models, corrected re-analyses overturning an ICLR 2025 Oral, a second case study targeting an ICLR 2026 Oral, and an operationalized checklist. Evaluation methodology papers are regularly accepted at top venues (Henderson 2018, AAAI; Dodge 2019, EMNLP; Dehghani 2021, NeurIPS).

**Compute cost.** The 6,000 A100-hour figure is the total sweep cost, not the protocol cost. The protocol itself is post-hoc subsampling from existing sweep results and adds zero compute. Even N=5 or N=10 per method can reveal whether a claimed advantage is robust, testable via a Mann-Whitney U-test.

**Presentation issues.** We will fix the typos, reference overflows, and oversized appendix figure. Thank you.
