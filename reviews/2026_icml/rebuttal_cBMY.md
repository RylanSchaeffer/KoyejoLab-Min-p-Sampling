# Response to Reviewer cBMY — ICML 2026 Submission 31762

**Status: DRAFT — needs review and polishing**

---

## General Response (to all reviewers)

Three of four reviewers raise the same concern: the blueprint relies on a single case study. We address this first, then note two cross-cutting revisions.

**Second case study: p-less sampling.** Since submission, we have applied Best-of-N to p-less sampling (Tan et al., ICLR 2026 Oral, arXiv:2509.23234v6), a truncation sampler claiming to "consistently outperform existing sampling approaches." We implemented p-less as a native sampler in vLLM v0.7.3 and are running sweeps across 28 models on GSM8K CoT, GSM8K CoT (Llama template), and GPQA — 5,952 runs on 3 seeds each. Preliminary results: [TBD: fill in when sweeps complete].

Independent of the Best-of-N results, the p-less paper already violates all four standards:
- **Standard 1 (Fair comparison):** Baselines use default hyperparameters. The paper's own Table 8 (Llama-2-7b only) shows tuned min-p matches or beats p-less on GPQA (0.249 vs 0.248). No tuned-baseline analysis exists for Mistral-7b or Llama3-70b.
- **Standard 2 (Valid inference):** No significance tests on any accuracy metric, despite including t-tests for efficiency claims (Table 14). Mistral-7B and Llama3-70B use only 1 random seed — making reported differences of 0.001 AUC uninterpretable.
- **Standard 3 (Transparency):** Human evaluation compares p-less at T=2.0 against default sampling at T=1.0, with 3 of 6 annotators being paper authors. No inter-annotator agreement reported. The Reproducibility Statement promises source code "upon publication," but the published repo contains only the sampler — no evaluation scripts, benchmarks, or sweep configurations.
- **Standard 4 (Consistent reporting):** "Consistently outperforms" is claimed despite min-p winning on 2/4 datasets for Llama3-70B (Table 1). The paper claims p-less "excels in the domain of creative writing," but at T=1.0 p-less ranks last among all 7 methods (Table 2). The AUC metric gives T=1.5 the single largest weight (33.3%) due to unevenly-spaced temperature points — the regime where p-less has its biggest advantage — never disclosed or justified.

Different paper, different authors, different venue, same evaluation problems. The blueprint generalizes.

**Independent corroboration: Artificial Hivemind (NeurIPS 2025 Best Paper).** Jiang et al. (2025) tested min-p for output diversity using aggressive settings (p=0.1, T=2.0) across 70+ models with 31,250 human annotations. They found 61% of response pairs exceeded 0.8 similarity — min-p does not meaningfully reduce mode collapse. This independently validates our finding that min-p's diversity claims are unsupported. Jiang et al. then concluded that "decoding-time interventions are fundamentally insufficient" — an over-generalization resting on the unverified assumption that min-p adequately represents decoding-time methods. Flawed upstream evaluation propagates incorrect conclusions downstream.

**Cross-cutting revisions.** Three reviewers (2LLS, dH2p, cBMY) note the paper reads as a critique of a specific work rather than a general framework. In the revision, the blueprint and Best-of-N protocol will be presented first as general tools, with the two case studies as illustrations. Detailed numerical discrepancies and author interactions will move to the appendix. Two reviewers (2LLS, dH2p) note the missing related work section; we will add one covering hyperparameter search bias (Dodge et al., 2019; Bouthillier et al., 2021), benchmarking fairness (Henderson et al., 2018; Dehghani et al., 2021), statistical evaluation (Dror et al., 2018; Pineau et al., 2021), and fair comparison precedents (Melis et al., 2020). We will also add Algorithm 1 (Best-of-N pseudocode) and an operationalized checklist validated against both case studies — see per-reviewer responses for details.

---

## Response to Reviewer cBMY

We address each concern below, beginning with the ethics flag.

### Ethics concerns

**On the paper reading as a "public rebuttal":** Scientific scrutiny of published work is foundational to the research process. Ioannidis (2005, "Why Most Published Research Findings Are False") is among the most cited papers in science; the NeurIPS Datasets & Benchmarks track explicitly encourages re-evaluation of existing claims. Our paper does not allege misconduct. It identifies methodological choices — unequal hyperparameter tuning, pooled statistics, selective reporting — that inflated reported results. These are evaluation methodology concerns, not integrity concerns. As Reviewer KrXT notes, the direct dialogue with Nguyen et al. "has already led to revisions to the investigated research" and "is a sign of productive and constructive scientific debate."

**On de-anonymization:** We acknowledge that including GitHub links created a de-anonymization risk and will replace all such links with anonymized references in the revision. We have filed a confidential AC comment regarding this matter.

### Q1: Best-of-N cost / Q2: Distinction from grid search

**Cost:** The 6,000 A100-hour figure is the cost of running hyperparameter sweeps, not the Best-of-N protocol itself. The protocol is trivial post-hoc computation on existing sweep data. The sweeps are the cost of fair comparison — a cost that should be paid whenever a paper claims one method outperforms another. The protocol degrades gracefully: even N=5-10 configurations per method reveals whether a claimed advantage is robust or fragile. We will demonstrate this empirically in the revision.

**Grid search distinction:** Grid search is an optimization algorithm: it selects the best hyperparameters for one method and outputs a single configuration. Best-of-N is an evaluation protocol: it diagnoses whether a claimed advantage survives equalized tuning budgets by producing comparative performance-vs-budget curves. Same mechanics, different purpose. We will add Algorithm 1 (pseudocode) in the revision.

### Q3: Single case study

We have added a second case study: p-less sampling (Tan et al., ICLR 2026 Oral). See the General Response. Different authors, different venue, different method — same evaluation methodology problems. Preliminary Best-of-N results are forthcoming; we will update once sweeps complete.

### Standards are known best practices / limited originality

These standards are individually well-known. But "well-known" and "well-practiced" are different. An ICLR 2025 Oral fails to follow these practices, and its central claims do not survive when the practices are applied. The p-less paper (ICLR 2026 Oral) independently exhibits the same violations (see General Response). If these standards were truly enforced, such papers would not receive oral designations at top venues.

The community has no systematic mechanism for post-publication empirical verification. Peer review checks methodology *descriptions* but rarely *reproduces* results. Without work like ours, violations surface only when researchers invest substantial effort to re-run experiments. Standards without enforcement remain aspirational.

Our contribution is: (1) formalizing Standard 1 into a reusable protocol (Best-of-N) with a specific diagnostic output, (2) demonstrating at scale that applying these principles overturns high-visibility claims, and (3) operationalizing all four standards into a concrete checklist validated against two independent case studies (see response to Reviewer 2LLS). As the reviewer acknowledges, several specific findings — on LLM-as-a-Judge under-specification, incorrect data pooling, and human evaluation annotations not supporting the claimed preference — are individually significant and would not exist without this work.

### Q4: How to enforce these standards

The reviewer correctly identifies that non-adoption reflects structural problems: publication incentives reward novelty over rigor, reviewer bandwidth limits methodological scrutiny, and a compute asymmetry exists between original authors (who run selective experiments) and verifiers (who must run comprehensive sweeps).

Precedent for progress exists: the ML Reproducibility Checklist (Pineau et al., 2021) was voluntarily adopted by NeurIPS, ICML, and ICLR and measurably improved reproducibility. Our operationalized checklist (see response to Reviewer 2LLS) could serve a similar role. Solving incentive problems requires community-level effort beyond a single paper, but concrete, validated tools are a necessary first step. We will add a discussion of adoption challenges to the Limitations section.

Our publicly released codebase, sweep data, and analysis notebooks lower the barrier for independent replication, which we actively encourage.

### Significance does not depend on min-p's significance

The reviewer suggests our significance "depends on the significance of min-p itself." The second case study (p-less) refutes this: the same blueprint, applied to a different method, reveals the same problems. The contribution is the diagnostic methodology and the empirical finding that high-visibility claims are fragile under rigorous evaluation — not the importance of any particular sampler.

### Presentation

The paper does not sufficiently explain min-p's motivations and mechanism, limiting accessibility. Min-p is an adaptive truncation sampler that removes tokens with probability below a fraction of the maximum token probability, intended as a temperature-robust alternative to top-p and top-k. We will add a self-contained "Background: Min-p Sampling" section in the revision.

We note that the reviewer was unfamiliar with min-p before reviewing and independently found the evidence on LLM-as-a-Judge under-specification, incorrect data pooling, and unsupported human evaluation claims persuasive.

Fixes planned: URL overflows, Figure 9 sizing, inline links moved to footnotes, reference format consistency, reduced repetition.

---

## Summary of Planned Revisions

For reference, here is what we plan for the camera-ready version:

1. **Second case study (p-less):** ~1 page demonstrating blueprint generality with new Best-of-N experiments across 28 models
2. **Algorithm 1:** Formal pseudocode for the Best-of-N evaluation protocol, with analysis of statistical properties (variance as a function of N, minimum N for reliable conclusions)
3. **Operationalized checklist:** Table mapping each standard to concrete items, failure modes, and violations found in both case studies
4. **Related Work section:** Positioning relative to Dodge 2019, Bouthillier 2021, Henderson 2018, Dror 2018, Melis 2020
5. **Background section:** What min-p is — motivation, mechanism, and claimed merits — for reader accessibility
6. **Grid search distinction:** Explicit paragraph differentiating Best-of-N (evaluation protocol) from grid search (optimization algorithm)
7. **Narrative restructuring:** Blueprint and protocol presented first as general tools; case studies reframed as illustrative applications; adversarial language softened; numerical discrepancies and author interactions moved to appendix
8. **Expanded Limitations section:** Adoption challenges, structural incentives, compute asymmetry between authors and verifiers
9. **Presentation fixes:** URL formatting, Figure 9 sizing, anonymized links, reference format consistency, reduced repetition
10. **Wilcoxon robustness checks:** Non-parametric confirmation of all statistical conclusions
