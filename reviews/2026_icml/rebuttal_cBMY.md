# Response to Reviewer cBMY

## General Response

Three of four reviewers note the blueprint relies on a single case study. We now have a second.

**Second case study: p-less sampling.** We applied Best-of-N to p-less (Tan et al., ICLR 2026 Oral), which claims to "consistently outperform." We implemented p-less in vLLM v0.7.3 and are sweeping 28 models on GSM8K CoT and GPQA (5,952 runs, 3 seeds). Best-of-N results will follow within 1-2 days.

P-less has a novel theoretical contribution (Renyi entropy) and real efficiency gains. Our critique targets evaluation, not the method. Independent of Best-of-N, it violates all four standards: default-only baselines (tuned min-p matches p-less on GPQA per their Table 8); no significance tests on accuracy despite having them for efficiency; human eval at T=2.0 vs T=1.0 with 3/6 author annotators; and "consistently outperforms" claimed when min-p wins 2/4 datasets on Llama3-70B, p-less loses 3/4 at T=1.0, and ranks last in creative writing at T=1.0. Different paper, authors, venue — same problems.

**Independent corroboration.** Jiang et al. (Artificial Hivemind, NeurIPS 2025 Best Paper) tested min-p for diversity across 70+ models: 61% of pairs exceeded 0.8 similarity. They over-generalized to "decoding-time interventions are insufficient" — based on the unverified assumption that min-p represents decoding-time methods.

**Revisions.** Blueprint and Best-of-N presented first; case studies as illustrations. Adding Related Work (Dodge 2019, Bouthillier 2021, Henderson 2018, Dehghani 2021, Dror 2018, Melis 2020), Algorithm 1 (pseudocode), and operationalized checklist.

---

## Response to Reviewer cBMY

**Ethics.** Our paper does not allege misconduct — it identifies evaluation methodology choices whose correction changes conclusions, in the tradition of Ioannidis (2005). All GitHub links will be anonymized in the revision.

**Q1: Cost.** The 6,000 A100-hours is the sweep cost, not the protocol cost. Best-of-N subsamples from existing results — zero additional compute. The sweeps are the cost of fair comparison. The protocol degrades gracefully: even N=5-10 reveals whether claims hold. Revision includes analysis of how conclusions stabilize at small N.

**Q2: Grid search distinction.** Grid search outputs one best configuration; Best-of-N outputs comparative performance-vs-budget curves diagnosing whether a claimed advantage survives equalized budgets. Like pass@k (Chen et al., 2021), the contribution is applying a known mechanism as a diagnostic protocol. Revision adds Algorithm 1.

**Q3: Single case study.** P-less (ICLR 2026 Oral) is our second case study — fundamentally different from min-p in design, yet exhibits the same four evaluation failures: default baselines (tuned min-p matches p-less on GPQA), no significance tests on accuracy, confounded human eval (3/6 author annotators, different temperatures), and overclaiming ("consistently outperforms" when min-p wins 2/4 on Llama3-70b). Sweeps across 28 models running; results within 1-2 days.

**Originality.** "Well-known" and "well-practiced" are different. Two ICLR Orals violate all four standards; their central claims do not survive rigorous application. Our contribution: (1) formalizing Standard 1 into Best-of-N with diagnostic output, (2) demonstrating at scale that applying these principles overturns high-visibility claims, (3) operationalizing all standards into a checklist validated against two papers.

**Q4: Enforcement.** Structural barriers are real. The ML Reproducibility Checklist (Pineau et al., 2021) shows adoption is possible. Our checklist targets the same path. Scaling enforcement through automated tools is active work in progress.

**Significance is not circular.** Our significance derives from downstream consequences, not min-p's importance. Contamination chain: Nguyen et al.'s flawed claims led Jiang et al. (NeurIPS 2025 Best Paper) to over-generalize that "decoding-time interventions are insufficient." Tan et al. (ICLR 2026 Oral) repeats the same flawed methodology. Three papers at top venues, one chain. The cost of absent standards is measurable in misdirected research.

**Presentation.** Revision adds min-p background section. Fixes: URL overflows, Figure 9 sizing, reference format, reduced repetition.
