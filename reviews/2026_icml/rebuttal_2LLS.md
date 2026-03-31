# Response to Reviewer 2LLS

## General Response

Three of four reviewers note that the blueprint relies on a single case study. Since submission, we have added a second: p-less sampling (Tan et al., ICLR 2026 Oral), a truncation sampler claiming to "consistently outperform existing sampling approaches." We implemented p-less in vLLM v0.7.3 and are running sweeps across 28 models (5,952 runs, 3 seeds each). Best-of-N results will follow within 1-2 days. P-less has a novel theoretical contribution (connecting thresholds to Renyi entropy) and real efficiency gains; our critique targets evaluation methodology, not the method itself.

Independent of Best-of-N, the p-less paper violates all four standards: baselines at default hyperparameters only (their own Table 8 shows tuned min-p matches p-less on GPQA); no significance tests on accuracy despite including them for efficiency; human evaluation comparing T=2.0 vs T=1.0 with 3 of 6 author annotators; and "consistently outperforms" claimed when min-p wins 2/4 datasets on Llama3-70B and p-less loses 3/4 at T=1.0. Different paper, different authors, different venue, same problems.

Separately, Jiang et al. (Artificial Hivemind, NeurIPS 2025 Best Paper) independently tested min-p for diversity across 70+ models and found 61% of response pairs exceeded 0.8 similarity, corroborating our finding that min-p's diversity claims are unsupported.

The revision will restructure the paper so the blueprint comes first, with case studies as illustrations. We are adding a Related Work section, Algorithm 1 (Best-of-N pseudocode), and an operationalized checklist validated against both case studies.

---

## Response to Reviewer 2LLS

**Q1: Code availability.** Yes. All code, sweep configurations, W&B sweep data, and analysis notebooks are publicly available at [anonymized URL]. The p-less vLLM patch will be included.

**Q2: Why equal-effort fairness?** The Best-of-N curve does not privilege one fairness philosophy. It reveals information relevant to several: "best achievable performance" (right end, large N), "tuning is part of the method" (left end, small N; faster rise means easier to tune), and "equal effort" (any fixed N). A single reported number collapses this into one point that can be gamed by choosing favorable hyperparameters. The curve makes the full picture visible. Limitation: effort is measured as configuration count, not compute; we discuss this in the revision.

**Best-of-N formalization.** Grid search selects the best hyperparameters for one method (output: one configuration). Best-of-N diagnoses whether a claimed advantage survives equalized tuning budgets (output: comparative performance-vs-budget curves). Same mechanics, different purpose. Like pass@k (Chen et al., 2021), the contribution is not the subsampling mechanism but its application as a diagnostic protocol. The revision adds Algorithm 1 (pseudocode) and analysis of how variance decreases with N.

**Related work.** Dodge et al. (2019) and Bouthillier et al. (2021) characterize unequal tuning but propose no reusable protocol. Henderson et al. (2018) show RL evaluation pitfalls without formalized comparison. Melis et al. (2020) do fair comparison ad hoc. Best-of-N systematizes these into a general diagnostic applicable post-hoc to existing sweep data.

**Standards 2-4 operationalization.** The revision adds a checklist mapping each standard to concrete items and failure modes, validated against both case studies. For example: Standard 2 requires per-model significance tests with correction (min-p pooled across models; p-less omitted tests entirely). Standard 3 requires releasing evaluation code (p-less released none). Standard 4 requires win/loss tables across all comparisons (both papers overclaimed). Scaling enforcement through automated tools is active work in progress.

**Tone and framing.** Blueprint and protocol will be presented first as general tools, with case studies as illustrations. Numerical discrepancies move to appendix; author interactions to footnotes.

**Venue fit.** The revised paper contains a formalized protocol with pseudocode, 6,000+ A100-hours of experiments across 28 models, corrected re-analyses overturning an ICLR 2025 Oral, a second case study (ICLR 2026 Oral), and an operationalized checklist. Evaluation methodology papers are regularly accepted at top venues: Henderson 2018 (AAAI), Dodge 2019 (EMNLP), Dehghani 2021 (NeurIPS).

**Cost.** The 6,000 A100-hour figure is the sweep cost, not the protocol cost. The protocol is post-hoc subsampling from existing results, adding zero compute. Even N=5-10 per method reveals whether a claimed advantage is robust.
