# Response to Reviewer cBMY

## General Response

Three reviewers note that the blueprint relies on a single case study. We have added a second: p-less sampling (Tan et al., ICLR 2026 Oral), which claims to "consistently outperform existing sampling approaches." We implemented p-less in vLLM v0.7.3 and are running sweeps across 28 models (5,952 runs, 3 seeds each). Best-of-N results will follow within 1-2 days.

P-less has a real theoretical contribution connecting thresholds to Renyi entropy, plus genuine efficiency gains. Our critique targets only evaluation methodology. The p-less paper violates all four standards: baselines at default hyperparameters only (their own Table 8 shows tuned min-p matches p-less on GPQA), no significance tests on accuracy, a human evaluation at different temperatures with 3 of 6 author annotators, and "consistently outperforms" claimed when min-p wins 2/4 datasets on Llama3-70B. Different paper, different authors, different venue, same problems.

Separately, Jiang et al. (NeurIPS 2025 Best Paper) independently tested min-p for diversity across 70+ models and found 61% of response pairs exceeded 0.8 similarity. This corroborates our finding that min-p's diversity claims are unsupported.

The revision restructures the paper so the blueprint comes first, with case studies as illustrations. We are adding a Related Work section, Algorithm 1 for Best-of-N, and a checklist validated against both case studies.

---

## Response to Reviewer cBMY

**Ethics.** Thank you for raising these concerns thoughtfully. We do not allege misconduct; we identify evaluation methodology choices whose correction changes the reported conclusions, following the tradition of Ioannidis (2005). We will anonymize all GitHub links in the revision.

**Q1: Isn't Best-of-N extremely costly?** The 6,000 A100-hour figure is the cost of running hyperparameter sweeps, not the Best-of-N protocol itself. Best-of-N subsamples from existing sweep results and adds zero compute. The sweeps are the cost of fair comparison, and that cost should be paid whenever a paper claims one method beats another. The protocol also degrades gracefully: even N=5 to 10 configurations per method reveals whether a claimed advantage holds. The revision will show how conclusions stabilize at small N.

**Q2: How does Best-of-N differ from grid search?** Grid search is an optimization tool: it finds the best hyperparameters for one method. Best-of-N is an evaluation tool: it asks whether a claimed advantage survives when every method gets the same tuning budget. The output is a comparative performance-vs-budget curve, not a single best configuration. The revision adds Algorithm 1 with pseudocode.

**Q3: Why only one case study?** P-less sampling (ICLR 2026 Oral) is now our second. It differs from min-p in design, yet shows the same four categories of evaluation failure: default-only baselines, missing significance tests, confounded human evaluation with author annotators, and overclaiming relative to its own tables. Sweeps across 28 models are running, with results expected within 1-2 days.

**Q4: How can these standards be enforced under existing incentives?** Structural barriers are real. But the ML Reproducibility Checklist (Pineau et al., 2021) shows that community adoption of evaluation standards is possible. Our checklist targets the same path. Scaling enforcement through automated tooling is work in progress.

**Originality.** "Well-known" and "well-practiced" are different things. Two ICLR Oral papers (2025 and 2026) violate all four standards, and the first paper's claims do not survive rigorous application. Our contributions: (1) formalizing fair comparison into a reusable protocol with diagnostic curves, (2) demonstrating at scale that applying these principles overturns high-visibility claims, and (3) operationalizing all four standards into a checklist validated against two independent papers.

**Significance is not circular.** Our significance comes from the downstream consequences of flawed evaluation, not from min-p's importance as a method. Nguyen et al.'s flawed claims led Jiang et al. (NeurIPS 2025 Best Paper) to over-generalize that "decoding-time interventions are insufficient." Tan et al. (ICLR 2026 Oral) repeated the same flawed methodology. Three papers at top venues, one contamination chain. The cost of absent evaluation standards is measurable in misdirected research.

**Presentation.** The revision adds a min-p background section for readers unfamiliar with the method. It also fixes URL overflows, Figure 9 sizing, and reference formatting.
