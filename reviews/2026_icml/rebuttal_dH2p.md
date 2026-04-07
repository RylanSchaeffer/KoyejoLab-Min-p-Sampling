# Response to Reviewer dH2p

## General Response

Three of four reviewers note that the blueprint relies on a single case study. We have since added a second: p-less sampling (Tan et al., ICLR 2026 Oral), a truncation sampler claiming to "consistently outperform existing sampling approaches." We implemented p-less in vLLM v0.7.3 and ran Best-of-N sweeps across 18 models on GSM8K and GPQA (5,022 runs, 3 seeds each). Under equal tuning budget, p-less's best configuration loses to the best other sampler on 40/45 model-benchmark pairs and wins on only 7/45.

The p-less paper violates all four standards independently of Best-of-N. Baselines use default hyperparameters only (their own Table 8 shows tuned min-p matches p-less on GPQA). Accuracy comparisons lack significance tests, even though the paper includes them for efficiency. Human evaluation compares T=2.0 vs. T=1.0 with 3 of 6 annotators being authors. The claim "consistently outperforms" does not hold: min-p wins 2/4 datasets on Llama3-70B, and p-less loses 3/4 at T=1.0. Different paper, different authors, different venue, same problems.

Separately, Jiang et al. (Artificial Hivemind, NeurIPS 2025 Best Paper) independently tested min-p for diversity across 70+ models. They found 61% of response pairs exceeded 0.8 similarity. This corroborates our finding that min-p's diversity claims are unsupported.

The revision restructures the paper so the blueprint comes first, with case studies as illustrations. We are also adding a Related Work section, Algorithm 1 (Best-of-N pseudocode), and an operationalized checklist validated against both case studies.

---

## Response to Reviewer dH2p

We thank the reviewer for rating significance as excellent. The core tension in the review is: significance 4/4, originality 1/4. We address this directly below.

**Empirical breadth.** We now have two case studies, and they are connected. Nguyen et al. (ICLR 2025 Oral) claimed min-p improves quality and diversity based on flawed evaluation. Jiang et al. (NeurIPS 2025 Best Paper) took this claim at face value and over-generalized to "decoding-time interventions are fundamentally insufficient." Tan et al. (ICLR 2026 Oral) then repeated the same flawed methodology for p-less. Three papers, three venues, three author groups, same evaluation failures. This is not a one-off problem.

**Originality.** The reviewer notes that the standards resemble widely discussed best practices. We agree they should be well-known. But two ICLR Oral papers violate all four. The gap between knowing and doing is exactly where this paper contributes.

Three specific contributions go beyond restating best practices. (1) Best-of-N is a diagnostic protocol, not a grid search. Grid search picks the best hyperparameters for one method. Best-of-N tests whether a claimed advantage survives equal tuning budget across methods, producing comparative curves. The revision adds Algorithm 1 with pseudocode. (2) We applied these principles at scale: over 6,000 A100-hours of experiments that overturned the central claims of two oral papers. (3) We provide an operationalized checklist validated against both case studies.

**Q1: Fair ranges for heterogeneous methods.** Best-of-N equalizes the configuration budget N, not the parameter space. Each method draws N configurations from its natural range. For example, p-less has 1 hyperparameter (temperature); min-p has 2 (temperature x min-p value). At N=20, p-less covers its space more densely. This asymmetry is conservative: it favors the method with more hyperparameters. When baselines matched min-p at equal N, min-p had this advantage and still did not outperform. If a method genuinely needs less tuning, Best-of-N shows this as a faster rise at low N.

**Q2: Venue fit.** Evaluation methodology papers are regularly accepted at top venues: Dodge 2019 (EMNLP), Dehghani 2021 (NeurIPS), Henderson 2018 (AAAI). None proposed new models. Our paper contributes a formalized protocol with pseudocode, over 6,000 A100-hours of experiments, corrected re-analyses that overturn an ICLR Oral, a second case study, and an operationalized checklist. If post-publication verification is excluded from main tracks, the field creates no venue or incentive for this work.

**Tone.** The revision restructures the paper: blueprint and protocol first, case studies as illustrations. Numerical discrepancies and author interactions move to the appendix.
