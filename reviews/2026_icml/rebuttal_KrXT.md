# Response to Reviewer KrXT — ICML 2026 Submission 31762

---

## General Response (to all reviewers)

Three of four reviewers raise the same concern: the blueprint relies on a single case study. We address this first, then describe cross-cutting revisions.

**Second case study: p-less sampling.** Since submission, we have applied Best-of-N to p-less sampling (Tan et al., ICLR 2026 Oral, arXiv:2509.23234v6), a truncation sampler claiming to "consistently outperform existing sampling approaches." We implemented p-less as a native sampler in vLLM v0.7.3 and are running sweeps across 28 models on GSM8K CoT, GSM8K CoT (Llama template), and GPQA — 5,952 runs on 3 seeds each. We will share Best-of-N results in a follow-up within 1-2 days.

P-less makes a genuinely novel theoretical contribution — connecting truncation thresholds to Renyi entropy — and has real efficiency advantages (O(|V|) complexity). Our critique targets evaluation methodology, not the method's design.

Independent of the Best-of-N results, the p-less paper violates all four standards:
- **Standard 1 (Fair comparison):** Baselines use default hyperparameters. The paper's own Table 8 (Llama-2-7b only) shows tuned min-p matches or beats p-less on GPQA (0.249 vs 0.248). No tuned-baseline analysis exists for Mistral-7b or Llama3-70b. Top-k is omitted entirely.
- **Standard 2 (Valid inference):** No significance tests on any accuracy metric, despite including t-tests for efficiency claims (Table 14). Mistral-7B and Llama3-70B use only 1 random seed — making reported differences of 0.001 AUC uninterpretable.
- **Standard 3 (Transparency):** Human evaluation compares p-less at T=2.0 against default sampling at T=1.0, with 3 of 6 annotators being paper authors. No inter-annotator agreement reported. The public repository contains only the sampler implementation — no evaluation scripts, benchmark code, or sweep configurations.
- **Standard 4 (Consistent reporting):** "Consistently outperforms" is claimed despite min-p winning on 2/4 datasets for Llama3-70B (Table 1). At T=1.0 — the most practically relevant temperature — p-less loses on 3/4 accuracy datasets for Llama3-70b (Table 5: epsilon 82.6 vs p-less 81.4 on CSQA; mirostat 41.1 vs p-less 38.4 on GPQA; min-p 90.6 vs p-less 89.8 on QASC). "Excels in creative writing" is claimed, but at T=1.0 p-less ranks last among all 7 methods (Table 2). The AUC metric gives T=1.5 the single largest weight (33.3%) due to unevenly-spaced temperature points — the regime where p-less has its biggest advantage — never disclosed or justified. The title claims a "hyperparameter-free" approach, but temperature is swept from 0.5 to 2.0.

Different paper, different authors, different venue, same evaluation problems. The blueprint generalizes.

**Independent corroboration: Artificial Hivemind (NeurIPS 2025 Best Paper).** Jiang et al. (2025) tested min-p for output diversity (p=0.1, T=2.0) across 70+ models with 31,250 human annotations and found 61% of response pairs exceeded 0.8 similarity — min-p does not meaningfully reduce mode collapse. This independently validates our finding that min-p's diversity claims are unsupported. Jiang et al. then concluded that "decoding-time interventions are fundamentally insufficient" — an over-generalization resting on the unverified assumption that min-p adequately represents decoding-time methods. Flawed upstream evaluation propagates incorrect conclusions downstream.

**Cross-cutting revisions.** The revision will restructure the paper so the blueprint and Best-of-N protocol come first as general tools, with the two case studies as illustrations. Detailed numerical discrepancies and author interactions move to the appendix. We add a Related Work section covering hyperparameter search bias (Dodge et al., 2019; Bouthillier et al., 2021), benchmarking fairness (Henderson et al., 2018; Dehghani et al., 2021), statistical evaluation (Dror et al., 2018; Pineau et al., 2021), and fair comparison precedents (Melis et al., 2020). We also add Algorithm 1 (Best-of-N pseudocode) and an operationalized checklist validated against both case studies — see per-reviewer responses for details.

---

## Response to Reviewer KrXT

We thank Reviewer KrXT for the careful evaluation.

### Q1: Statistical test selection and justification

Each choice follows from the data structure and the claim under test:

- **Paired t-tests:** For each (model, temperature, seed) triple, every sampler is evaluated on the same prompts. Pairing controls for model-level and prompt-level variance, yielding tighter confidence intervals than unpaired alternatives.
- **One-sided tests:** The claim is directional ("min-p outperforms"), so we test H_a: min-p > other. Failure to reject H_0 means the claimed advantage is unsupported.
- **Bonferroni correction:** The most conservative standard correction. We deliberately chose Bonferroni over less conservative alternatives (Holm, BH) so that any rejection is robust to methodological criticism. Result: Nguyen et al. claimed min-p dominates across all 12 (model, benchmark) settings. Under Bonferroni correction, only 1 of 12 rejects the null.
- **Intersection-Union Test (IUT):** "Consistent superiority" means the effect holds in every setting simultaneously — exactly IUT's alternative hypothesis. The null is the union of "min-p does not outperform in setting x"; the alternative is the intersection of "min-p outperforms in all settings." We fail to reject.

The revision will include Wilcoxon signed-rank tests as a non-parametric robustness check, removing any dependence on distributional assumptions.

### Q2: Best-of-N vs prior work (e.g., Mogrifier LSTM)

Best-of-N detects cherry-picking by equalizing hyperparameter budget across methods. In Nguyen et al., min-p received a more extensive hyperparameter sweep than top-p, making it appear superior. Best-of-N removes this confound.

Melis et al. (2020) demonstrated the same core insight — fair hyperparameter comparison reverses published rankings — but their analysis was ad hoc and setting-specific. Best-of-N formalizes this into a reusable protocol with a standard diagnostic output (performance-vs-budget curves) and pseudocode (Algorithm 1 in the revision). We cite Melis et al. in the new Related Work section.

The distinction from grid search: grid search optimizes hyperparameters for a single method (output: one configuration); Best-of-N diagnoses whether a claimed advantage survives equalized tuning effort (output: comparative curves across methods). Like pass@k in code generation (Chen et al., 2021), the contribution is not the subsampling mechanism but its application as a diagnostic protocol answering a specific question: does the advantage persist when optimization budgets are equalized?

**On the concern that "fair comparison through hyperparameter search is the main driving factor."** This is exactly right — and it is the finding, not a limitation. The Best-of-N curves show baselines match or exceed min-p at equal budgets. The original advantage was an artifact of unequal tuning. That equalization is "the main driving factor" confirms the protocol's diagnostic value: it identifies the specific mechanism by which claims were inflated. The second case study (p-less; see General Response) shows the same pattern in a different paper, by different authors, at a different venue.

---

## Summary of Planned Revisions

Key changes for the camera-ready, beyond the General Response:

1. **Algorithm 1:** Formal pseudocode for Best-of-N, with variance analysis as a function of N and minimum N for reliable conclusions
2. **Wilcoxon robustness checks:** Non-parametric tests alongside existing t-tests
3. **Grid search distinction:** Explicit paragraph differentiating Best-of-N (diagnostic evaluation protocol) from grid search (optimization algorithm), with pass@k analogy
4. **Operationalized checklist:** Table mapping each standard to concrete items, failure modes, and violations found in both case studies
5. **Expanded Limitations section:** Adoption challenges, structural incentives, compute asymmetry between authors and verifiers
6. **Presentation fixes:** URL formatting, Figure 9 sizing, anonymized links, reference format consistency
