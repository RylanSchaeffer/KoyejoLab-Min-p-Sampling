# Response to Reviewer KrXT — ICML 2026 Submission 31762

**Status: DRAFT — needs review and polishing**

---

## General Response (to all reviewers)

Three of four reviewers raise the same concern: the blueprint relies on a single case study. We address this first, then note two cross-cutting revisions.

**Second case study: p-less sampling.** Since submission, we have applied Best-of-N to p-less sampling (Tan et al., ICLR 2026 Oral, arXiv:2509.23234v6), a truncation sampler claiming to "consistently outperform existing sampling approaches." We implemented p-less as a native sampler in vLLM v0.7.3 and are running sweeps across 28 models on GSM8K CoT, GSM8K CoT (Llama template), and GPQA — 5,952 runs on 3 seeds each. Preliminary results: [TBD: fill in when sweeps complete].

Independent of the Best-of-N results, the p-less paper already violates all four standards:
- **Standard 1 (Fair comparison):** Baselines use default hyperparameters. The paper's own Table 8 (Llama-2-7b only) shows tuned min-p matches or beats p-less on GPQA (0.249 vs 0.248). No tuned-baseline analysis exists for Mistral-7b or Llama3-70b.
- **Standard 2 (Valid inference):** No significance tests on any accuracy metric, despite including t-tests for efficiency claims (Table 14). Mistral-7B and Llama3-70B use only 1 random seed — making reported differences of 0.001 AUC uninterpretable.
- **Standard 3 (Transparency):** Human evaluation compares p-less at T=2.0 against default sampling at T=1.0, with 3 of 6 annotators being paper authors. No inter-annotator agreement reported. No evaluation code released.
- **Standard 4 (Consistent reporting):** "Consistently outperforms" is claimed despite min-p winning on 2/4 datasets for Llama3-70B (Table 1). The AUC metric assigns 2/3 of its weight to temperatures above 1.0, inflating p-less's high-temperature advantage — never discussed or justified.

Different paper, different authors, different venue, same evaluation problems. The blueprint generalizes.

**Independent corroboration: Artificial Hivemind (NeurIPS 2025 Best Paper).** Jiang et al. (2025) tested min-p for output diversity using aggressive settings (p=0.1, T=2.0) across 70+ models with 31,250 human annotations. They found 61% of response pairs exceeded 0.8 similarity — min-p does not meaningfully reduce mode collapse. This independently validates our finding that min-p's diversity claims are unsupported. Jiang et al. then concluded that "decoding-time interventions are fundamentally insufficient" — an over-generalization resting on the unverified assumption that min-p adequately represents decoding-time methods. Flawed upstream evaluation propagates incorrect conclusions downstream.

**Cross-cutting revisions.** Three reviewers (2LLS, dH2p, cBMY) note the paper reads as a critique of a specific work rather than a general framework. In the revision, the blueprint and Best-of-N protocol will be presented first as general tools, with the two case studies as illustrations. Detailed numerical discrepancies and author interactions will move to the appendix. Two reviewers (2LLS, dH2p) note the missing related work section; we will add one covering hyperparameter search bias (Dodge et al., 2019; Bouthillier et al., 2021), benchmarking fairness (Henderson et al., 2018; Dehghani et al., 2021), statistical evaluation (Dror et al., 2018; Pineau et al., 2021), and fair comparison precedents (Melis et al., 2020). We will also add Algorithm 1 (Best-of-N pseudocode) and an operationalized checklist validated against both case studies — see per-reviewer responses for details.

---

## Response to Reviewer KrXT

We thank Reviewer KrXT for the thorough review.

### Q1: Statistical test selection and justification

Our statistical framework:

- **Paired t-tests:** Observations are naturally paired — for each (model, temperature, seed) triple, we have scores from each sampler on the same evaluation set. Pairing controls for model-level and prompt-level variance.
- **One-sided tests:** The claim under investigation is directional ("min-p outperforms other samplers"). We test H_a: min-p > other, so *failure* to reject H_0 means we cannot confirm the claimed advantage.
- **Bonferroni correction:** The most conservative multiple-testing adjustment. We prefer Bonferroni over less conservative alternatives (Holm, BH) so that any significant results are robust to criticism.
- **Intersection-Union Test (IUT):** The original claim is "consistent superiority" — min-p should outperform across all models. IUT requires the effect to hold in every subgroup simultaneously, which is the formal translation of "consistently outperforms."

As a robustness check, Wilcoxon signed-rank tests (no distributional assumptions) yield identical conclusions on all comparisons. We will add this detail to the revision.

### Q2: Best-of-N vs prior work (e.g., Mogrifier LSTM)

Prior work like Melis et al. (2020, Mogrifier LSTM) demonstrated the importance of fair hyperparameter comparison — but did so ad hoc, with effort specific to their setting. The insight is correct; the reusable tool is missing. We will cite this connection in our Related Work section.

Best-of-N formalizes this into a protocol with a standard diagnostic output: comparative performance-vs-budget curves. The distinction from grid search:

- **Grid search** selects the best hyperparameters for one method. Output: a single configuration.
- **Best-of-N** diagnoses whether a claimed advantage is real or an artifact of unequal tuning. Output: curves showing relative performance as a function of hyperparameter budget across methods.

Crucially, the Best-of-N curves do not merely show that "more search helps everything equally." They show that baselines *match or exceed* min-p at equal budgets — meaning the original advantage was specifically an artifact of unequal tuning, not a genuine algorithmic improvement. This is a substantive empirical finding, not a foregone conclusion.

We will add Algorithm 1 (pseudocode) to the revision.

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
