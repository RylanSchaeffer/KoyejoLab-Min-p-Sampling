# Response to Reviewer dH2p

## General Response

Three of four reviewers note that the blueprint relies on a single case study. Since submission, we have added a second: p-less sampling (Tan et al., ICLR 2026 Oral), a truncation sampler claiming to "consistently outperform existing sampling approaches." We implemented p-less in vLLM v0.7.3 and are running sweeps across 28 models (5,952 runs, 3 seeds each). Best-of-N results will follow within 1-2 days. P-less has a novel theoretical contribution (connecting thresholds to Renyi entropy) and real efficiency gains; our critique targets evaluation methodology, not the method itself.

Independent of Best-of-N, the p-less paper violates all four standards: baselines at default hyperparameters only (their own Table 8 shows tuned min-p matches p-less on GPQA); no significance tests on accuracy despite including them for efficiency; human evaluation comparing T=2.0 vs T=1.0 with 3 of 6 author annotators; and "consistently outperforms" claimed when min-p wins 2/4 datasets on Llama3-70B and p-less loses 3/4 at T=1.0. Different paper, different authors, different venue, same problems.

Separately, Jiang et al. (Artificial Hivemind, NeurIPS 2025 Best Paper) independently tested min-p for diversity across 70+ models and found 61% of response pairs exceeded 0.8 similarity, corroborating our finding that min-p's diversity claims are unsupported.

The revision will restructure the paper so the blueprint comes first, with case studies as illustrations. We are adding a Related Work section, Algorithm 1 (Best-of-N pseudocode), and an operationalized checklist validated against both case studies.

---

## Response to Reviewer dH2p

The reviewer rates significance as excellent (4/4) but originality as poor (1/4). We take this tension seriously: the contribution is not a new algorithm but a diagnostic protocol and the empirical infrastructure to overturn incorrect claims at scale.

**Empirical breadth.** The two case studies trace a contamination chain. Nguyen et al. (ICLR 2025 Oral) claimed min-p improves quality and diversity based on flawed evaluation. Jiang et al. (NeurIPS 2025 Best Paper) took this at face value and over-generalized to "decoding-time interventions are fundamentally insufficient." Tan et al. (ICLR 2026 Oral, p-less) repeated the same flawed methodology. Three papers, three venues, three author groups, same evaluation failures.

**Originality.** If these standards are well-known, why do two ICLR Oral papers violate all four? The gap between knowing best practices and systematically applying them is where the contribution lies. First, Best-of-N as a diagnostic protocol: like pass@k (Chen et al., 2021), the subsampling mechanism is well-known, but repurposing it to answer "does the advantage survive equalized budgets?" is new. Grid search outputs one configuration; Best-of-N outputs comparative curves. The revision adds Algorithm 1. Second, empirical verification at scale: applying these principles required >6,000 A100-hours and overturned the central claims of two oral papers. Third, an operationalized checklist validated against both case studies.

**Q1: Fair ranges for heterogeneous methods.** Best-of-N equalizes the configuration budget N, not the parameter space. Each method draws N configurations from its natural space. P-less has 1 hyperparameter (temperature); min-p has 2 (temperature x min-p value). At N=20, p-less covers its space more densely, an asymmetry that is conservative: it favors the richer-parameter method. When baselines match min-p at equal N, min-p had the advantage and still didn't outperform. If a method genuinely needs less tuning, this shows in the curve as faster rise at low N.

**Q2: Venue fit.** Evaluation methodology papers are regularly accepted at top venues: Dodge 2019 (EMNLP), Dehghani 2021 (NeurIPS), Henderson 2018 (AAAI), none of which proposed new models. Our paper adds a formalized protocol with pseudocode, 6,000+ A100-hours of experiments, corrected re-analyses overturning an ICLR Oral, a second case study, and an operationalized checklist. If post-publication verification is excluded from main tracks, the field provides no venue or incentive for this work.

**Tone.** The revision restructures the paper: blueprint and protocol first, case studies as illustrations. Numerical discrepancies and author interactions move to the appendix.
