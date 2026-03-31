# Response to Reviewer cBMY — ICML 2026 Submission 31762

---

## General Response (to all reviewers)

Three of four reviewers raise the same concern: the blueprint relies on a single case study. We now have a second, and describe cross-cutting revisions below.

**Second case study: p-less sampling.** Since submission, we have applied Best-of-N to p-less sampling (Tan et al., ICLR 2026 Oral, arXiv:2509.23234v6), a truncation sampler claiming to "consistently outperform existing sampling approaches." We implemented p-less as a native sampler in vLLM v0.7.3 and are running sweeps across 28 models on GSM8K CoT, GSM8K CoT (Llama template), and GPQA — 5,952 runs on 3 seeds each. We will share Best-of-N results in a follow-up within 1-2 days.

P-less makes a genuinely novel theoretical contribution — connecting truncation thresholds to Renyi entropy — and has real efficiency advantages (O(|V|) complexity). Our critique targets evaluation methodology, not the method's design.

Independent of the Best-of-N results, the p-less paper violates all four standards. Standard 1 (Fair comparison): baselines use default hyperparameters; the paper's own Table 8 (Llama-2-7b only) shows tuned min-p matches or beats p-less on GPQA (0.249 vs 0.248), and no tuned-baseline analysis exists for Mistral-7b or Llama3-70b; top-k is omitted entirely. Standard 2 (Valid inference): no significance tests on any accuracy metric, despite including t-tests for efficiency claims (Table 14); Mistral-7B and Llama3-70B use only 1 random seed, making reported differences of 0.001 AUC uninterpretable. Standard 3 (Transparency): human evaluation compares p-less at T=2.0 against default sampling at T=1.0, with 3 of 6 annotators being paper authors and no inter-annotator agreement reported; the public repository contains only the sampler implementation — no evaluation scripts, benchmark code, or sweep configurations. Standard 4 (Consistent reporting): "consistently outperforms" is claimed despite min-p winning on 2/4 datasets for Llama3-70B (Table 1); at T=1.0, p-less loses on 3/4 accuracy datasets for Llama3-70b (Table 5: epsilon 82.6 vs p-less 81.4 on CSQA; mirostat 41.1 vs p-less 38.4 on GPQA; min-p 90.6 vs p-less 89.8 on QASC); "excels in creative writing" is claimed, but at T=1.0 p-less ranks last among all 7 methods (Table 2); the AUC metric gives T=1.5 the single largest weight (33.3%) due to unevenly-spaced temperature points — the regime where p-less has its biggest advantage — never disclosed or justified; the title claims a "hyperparameter-free" approach, but temperature is swept from 0.5 to 2.0.

Different paper, different authors, different venue, same evaluation problems. The blueprint generalizes.

**Independent corroboration: Artificial Hivemind (NeurIPS 2025 Best Paper).** Jiang et al. (2025) tested min-p for output diversity (p=0.1, T=2.0) across 70+ models with 31,250 human annotations and found 61% of response pairs exceeded 0.8 similarity — min-p does not meaningfully reduce mode collapse. This independently validates our finding that min-p's diversity claims are unsupported. Jiang et al. then concluded that "decoding-time interventions are fundamentally insufficient" — an over-generalization resting on the unverified assumption that min-p adequately represents decoding-time methods. Flawed upstream evaluation propagates incorrect conclusions downstream.

**Cross-cutting revisions.** The revision restructures the paper so the blueprint and Best-of-N protocol come first as general tools, with the two case studies as illustrations. Detailed numerical discrepancies and author interactions move to the appendix. New additions: a Related Work section covering hyperparameter search bias (Dodge et al., 2019; Bouthillier et al., 2021), benchmarking fairness (Henderson et al., 2018; Dehghani et al., 2021), statistical evaluation (Dror et al., 2018; Pineau et al., 2021), and fair comparison precedents (Melis et al., 2020); Algorithm 1 (Best-of-N pseudocode); and an operationalized checklist validated against both case studies — see per-reviewer responses for details.

---

## Response to Reviewer cBMY

### Ethics concerns

We appreciate the careful flagging. Our paper does not allege misconduct — it identifies evaluation methodology choices (unequal tuning, pooled statistics, selective reporting) whose correction changes the reported conclusions, squarely within the tradition of Ioannidis (2005) and the NeurIPS Datasets & Benchmarks track. We will replace all GitHub links with anonymized references in the revision.

### Q1: Best-of-N cost / Q2: Distinction from grid search

**Cost.** The 6,000 A100-hour figure reflects running comprehensive hyperparameter sweeps across all models and samplers, not the Best-of-N protocol itself. The protocol subsamples from existing sweep results and adds zero additional compute. The sweeps are the cost of fair comparison — a cost that should be paid whenever a paper claims one method outperforms another. The protocol also degrades gracefully: even N=5-10 configurations per method reveals whether a claimed advantage is robust or fragile. The revision will include analysis showing how conclusions stabilize at small N.

**Grid search distinction.** Grid search is an optimization algorithm that outputs a single best configuration. Best-of-N is an evaluation protocol that diagnoses whether a claimed advantage survives equalized tuning budgets by producing comparative performance-vs-budget curves. Same mechanics, different purpose — analogous to how pass@k (Chen et al., 2021) uses well-known subsampling but as a diagnostic evaluation metric, or how Yang et al. (2025) repurposed pass@k to ask whether RL teaches new reasoning. The novelty is the question and the finding, not the mechanism. The revision will include Algorithm 1 (pseudocode).

### Q3: Single case study

We have added a second case study: p-less sampling (Tan et al., ICLR 2026 Oral), an information-theoretic truncation sampler fundamentally different from min-p in design and motivation. P-less independently exhibits the same four categories of evaluation failure: default-only baselines (the paper's own Table 8 shows tuned min-p matches p-less on GPQA), no significance tests on accuracy (despite including them for efficiency), a confounded human evaluation (different temperatures, 3 of 6 annotators are paper authors), and overclaiming ("consistently outperforms" when min-p wins 2/4 datasets on Llama3-70b; "excels in creative writing" when p-less ranks last at T=1.0). Different authors, different venue, different method — same problems. Sweeps across 28 models are running; Best-of-N results will follow within 1-2 days.

### Standards are known best practices / limited originality

"Well-known" and "well-practiced" are different. Two ICLR Oral papers (2025, 2026) fail to follow these standards, and the central claims of the first do not survive when they are applied. If these standards were truly enforced, such papers would not receive oral designations at top venues.

Our contribution is threefold: (1) formalizing Standard 1 into a reusable protocol (Best-of-N) with a specific diagnostic output, (2) demonstrating at scale that applying these principles overturns high-visibility claims, and (3) operationalizing all four standards into a concrete checklist validated against two independent case studies (see response to Reviewer 2LLS). As the reviewer acknowledges, the specific findings — on LLM-as-a-Judge under-specification, incorrect data pooling, and human evaluation annotations contradicting the claimed preference — are individually significant and would not exist without this work.

### Q4: How to enforce these standards

The reviewer correctly identifies structural barriers: incentives reward novelty over rigor, reviewer bandwidth limits methodological scrutiny, and a compute asymmetry exists between original authors and verifiers. Precedent for progress exists: the ML Reproducibility Checklist (Pineau et al., 2021) was voluntarily adopted by NeurIPS, ICML, and ICLR and measurably improved reproducibility. Our operationalized checklist (see response to Reviewer 2LLS) targets the same adoption path. Scaling enforcement through automated evaluation tools is active work in progress. The revision will add a discussion of adoption challenges to the Limitations section; our publicly released codebase, sweep data, and analysis notebooks already lower the barrier for independent verification.

### Significance does not depend on min-p's significance

The reviewer suggests our significance "depends on the significance of min-p itself." The opposite is true: our significance derives from the downstream consequences of flawed evaluation, not from min-p as a method. A concrete contamination chain demonstrates this. Nguyen et al.'s flawed claims are taken at face value by Jiang et al. (Artificial Hivemind, NeurIPS 2025 Best Paper), who test min-p for diversity, get a null result, and over-generalize to "decoding-time interventions are fundamentally insufficient." The premise was wrong — min-p was never genuinely shown to increase diversity. Meanwhile, p-less (ICLR 2026 Oral) repeats the same evaluation methodology our blueprint identifies as flawed. Three papers at top venues, one contamination chain. The cost of not adopting rigorous evaluation standards is measurable in misdirected research effort. The second case study independently confirms this: same blueprint, different method, same problems. The contribution is the diagnostic methodology and the finding that high-visibility claims are fragile under rigorous evaluation — not the importance of any particular sampler.

### Presentation

The revision will add a "Background: Min-p Sampling" section explaining min-p as an adaptive truncation sampler that removes tokens with probability below a fraction of the maximum token probability. Additional fixes: URL overflows, Figure 9 sizing, inline links moved to footnotes, reference format consistency, reduced repetition.

---

## Summary of Planned Revisions

For reference, here is what we plan for the camera-ready version:

1. **Second case study (p-less):** ~1 page demonstrating blueprint generality with new Best-of-N experiments across 28 models
2. **Algorithm 1:** Formal pseudocode for the Best-of-N evaluation protocol, with analysis of statistical properties (variance as a function of N, minimum N for reliable conclusions)
3. **Operationalized checklist:** Table mapping each standard to concrete items, failure modes, and violations found in both case studies
4. **Related Work section:** Positioning relative to Dodge 2019, Bouthillier 2021, Henderson 2018, Dror 2018, Melis 2020
5. **Background section:** What min-p is — motivation, mechanism, and claimed merits — for reader accessibility
6. **Grid search distinction:** Explicit paragraph differentiating Best-of-N (evaluation protocol) from grid search (optimization algorithm), with pass@k analogy
7. **Contamination chain discussion:** How flawed evaluation propagates across papers (Nguyen et al. -> Artificial Hivemind -> p-less) in the Discussion section
8. **Narrative restructuring:** Blueprint and protocol presented first as general tools; case studies reframed as illustrative applications; adversarial language softened; numerical discrepancies and author interactions moved to appendix
9. **Expanded Limitations section:** Adoption challenges, structural incentives, compute asymmetry between authors and verifiers
10. **Presentation fixes:** URL formatting, Figure 9 sizing, anonymized links, reference format consistency, reduced repetition
11. **Wilcoxon robustness checks:** Non-parametric confirmation of all statistical conclusions
