# Response to Reviewer cBMY

## General Response

Three of four reviewers note that the blueprint relies on a single case study. Since submission, we have added a second: p-less sampling (Tan et al., ICLR 2026 Oral), a truncation sampler claiming to "consistently outperform existing sampling approaches." We implemented p-less in vLLM v0.7.3 and are running sweeps across 28 models (5,952 runs, 3 seeds each). Best-of-N results will follow within 1-2 days. P-less has a novel theoretical contribution (connecting thresholds to Renyi entropy) and real efficiency gains; our critique targets evaluation methodology, not the method itself.

Independent of Best-of-N, the p-less paper violates all four standards: baselines at default hyperparameters only (their own Table 8 shows tuned min-p matches p-less on GPQA); no significance tests on accuracy despite including them for efficiency; human evaluation comparing T=2.0 vs T=1.0 with 3 of 6 author annotators; and "consistently outperforms" claimed when min-p wins 2/4 datasets on Llama3-70B and p-less loses 3/4 at T=1.0. Different paper, different authors, different venue, same problems.

Separately, Jiang et al. (Artificial Hivemind, NeurIPS 2025 Best Paper) independently tested min-p for diversity across 70+ models and found 61% of response pairs exceeded 0.8 similarity, corroborating our finding that min-p's diversity claims are unsupported.

The revision will restructure the paper so the blueprint comes first, with case studies as illustrations. We are adding a Related Work section, Algorithm 1 (Best-of-N pseudocode), and an operationalized checklist validated against both case studies.

---

## Response to Reviewer cBMY

**Ethics.** Thank you for raising these concerns carefully. Our paper does not allege misconduct. It identifies evaluation methodology choices (unequal tuning, pooled statistics, selective reporting) whose correction changes the reported conclusions, in the tradition of Ioannidis (2005) and the NeurIPS Datasets & Benchmarks track. We will anonymize all GitHub links in the revision.

**Q1: Cost.** The 6,000 A100-hour figure is the cost of running hyperparameter sweeps, not the Best-of-N protocol itself. The protocol subsamples from existing sweep results and adds zero compute. The sweeps are the cost of fair comparison, which should be paid whenever a paper claims one method outperforms another. The protocol degrades gracefully: even N=5-10 configurations per method reveals whether a claimed advantage is robust. The revision will include analysis showing how conclusions stabilize at small N.

**Q2: Grid search distinction.** Grid search is an optimization protocol: it picks the best hyperparameters for one method. Best-of-N is an evaluation protocol: it tests whether a claimed advantage survives when all methods receive equal tuning budget, producing comparative performance-vs-budget curves. The revision adds Algorithm 1 (pseudocode).

**Q3: Single case study.** P-less (ICLR 2026 Oral) is our second case study. It is fundamentally different from min-p in design, yet exhibits the same four categories of evaluation failure: default-only baselines, no significance tests on accuracy, a confounded human evaluation with author annotators, and overclaiming relative to its own tables. Sweeps across 28 models are running; results within 1-2 days.

**Originality.** "Well-known" and "well-practiced" are different. Two ICLR Oral papers (2025, 2026) violate all four standards, and the claims of the first do not survive rigorous application. Our contribution: (1) formalizing Standard 1 into a reusable protocol with diagnostic output, (2) demonstrating at scale that applying these principles overturns high-visibility claims, (3) operationalizing all standards into a checklist validated against two independent papers.

**Q4: Enforcement.** Structural barriers are real. The ML Reproducibility Checklist (Pineau et al., 2021) shows adoption is possible; our checklist targets the same path. Scaling enforcement through automated tools is active work in progress.

**Significance is not circular.** Our significance derives from the downstream consequences of flawed evaluation, not from min-p's importance as a method. Nguyen et al.'s flawed claims led Jiang et al. (NeurIPS 2025 Best Paper) to over-generalize that "decoding-time interventions are insufficient," and Tan et al. (ICLR 2026 Oral) repeated the same flawed methodology. Three papers at top venues, one contamination chain. The cost of absent evaluation standards is measurable in misdirected research.

**Presentation.** The revision adds a min-p background section and fixes URL overflows, Figure 9 sizing, and reference formatting.
