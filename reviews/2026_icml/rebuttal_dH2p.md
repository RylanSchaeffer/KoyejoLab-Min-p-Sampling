# Response to Reviewer dH2p — ICML 2026 Submission 31762

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

## Response to Reviewer dH2p

The concerns center on empirical breadth, originality, and presentation. We address each below.

### Limited empirical breadth

We have added a second case study: p-less sampling (Tan et al., ICLR 2026 Oral). See the General Response. Different paper, different authors, different venue — same evaluation methodology problems under our blueprint. [TBD: preliminary Best-of-N results show the same pattern.]

The contamination chain reinforces the urgency. Nguyen et al. (2025) claimed min-p improves quality and diversity based on flawed evaluation. Jiang et al. (2025, Artificial Hivemind, NeurIPS Best Paper) took this at face value, tested min-p for diversity, and over-generalized to "decoding-time interventions are fundamentally insufficient." Tan et al. (2026, p-less) proposed a new sampler using the same flawed methodology. Without tools to distinguish genuine advances from tuning artifacts, the field cycles through methods while propagating upstream errors downstream. Our blueprint provides those tools.

### Unclear methodological contribution beyond best practices

Fair comparison, valid statistics, transparency, and consistent reporting are individually well-known. Our contribution is not inventing these principles but:

1. **Formalizing Standard 1** into the Best-of-N protocol — a reusable diagnostic tool that produces comparative performance-vs-budget curves. Grid search selects hyperparameters for one method; Best-of-N diagnoses whether a claimed advantage survives equalized tuning budgets across methods. Algorithm 1 (pseudocode) will be added to the revision.

2. **Demonstrating at scale** that applying these principles overturns the central claims of two oral papers at top venues (ICLR 2025, ICLR 2026). If these principles were truly followed in practice, such papers would not pass peer review with oral designations. The gap between knowing best practices and enforcing them is the problem. The community has no systematic mechanism for post-publication empirical verification — these violations only surface when researchers invest substantial effort to reproduce and scrutinize, which is what our paper does.

3. **Operationalizing all four standards** into a concrete checklist (see our response to Reviewer 2LLS) validated against two independent case studies.

### Tone and framing

In the revision, the blueprint and Best-of-N protocol will be presented first as general tools, with case studies serving as illustrations rather than driving the narrative. Detailed numerical discrepancies and author interactions will move to the appendix. Language will be revised for consistency with constructive scientific discourse.

### Q1: Fair hyperparameter ranges for heterogeneous methods

Best-of-N equalizes the *budget* (number of configurations N), not the search space. Each method draws N configurations from whatever parameter space is natural for it. A method with one hyperparameter (e.g., p-less: temperature only) has its space covered more densely at a given N than a method with two (e.g., min-p: temperature x min-p value). This is a real asymmetry, but it cuts *against* finding that simpler methods match complex ones — so when Best-of-N shows that temperature-only baselines match min-p at equal N, the conclusion is conservative. If anything, min-p had an advantage from its richer parameter space and still failed to outperform.

This is one of Best-of-N's advantages over fixed-grid comparisons: it handles methods with qualitatively different parameter spaces without requiring ad hoc decisions about "equivalent" search ranges.

### Q2: Venue fit

See our response to Reviewer 2LLS. The submission contains: (1) a formalized evaluation protocol with pseudocode, (2) over 6,000 A100-hours of new experiments across 28 models, (3) corrected re-analyses showing a high-visibility ICLR 2025 Oral's claims are unsupported, (4) a second case study demonstrating generality, and (5) an operationalized checklist validated against two independent papers. Henderson et al. (2018, "Deep RL that Matters") was published at AAAI with a similar structure. If post-publication verification is excluded from main research tracks, the field has no venue — and therefore no incentive — for this work. The cost is that evaluation failures in high-visibility papers go uncorrected and propagate downstream, as we document.

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
