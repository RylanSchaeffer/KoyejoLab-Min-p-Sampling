# Response to Reviewer dH2p — ICML 2026 Submission 31762

---

## General Response (to all reviewers)

All three weak-reject/reject reviewers raise the same concern: the blueprint relies on a single case study. We address this first, then describe revisions.

**Second case study: p-less sampling.** Since submission, we have applied Best-of-N to p-less sampling (Tan et al., ICLR 2026 Oral, arXiv:2509.23234v6), a truncation sampler claiming to "consistently outperform existing sampling approaches." We implemented p-less as a native sampler in vLLM v0.7.3 and are running sweeps across 28 models on GSM8K CoT, GSM8K CoT (Llama template), and GPQA — 5,952 runs on 3 seeds each. Results will follow within 1-2 days.

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

## Response to Reviewer dH2p

The reviewer rates significance as excellent (4/4) but originality as poor (1/4). We take this tension seriously: the contribution is not a new algorithm but a diagnostic protocol and the empirical infrastructure to overturn incorrect claims at scale. We address each concern directly.

### Limited empirical breadth

The revision includes a second case study: p-less sampling (Tan et al., ICLR 2026 Oral), an information-theoretic truncation sampler with no design relationship to min-p. Applying our blueprint to p-less reveals all four categories of violation: baselines evaluated at default hyperparameters only, no significance tests on accuracy, a human evaluation comparing methods at different temperatures with author annotators, and claims contradicted by the paper's own tables ("consistently outperforms" when min-p wins 2/4 datasets on Llama3-70B). Best-of-N sweeps across 28 models with 3 seeds each are running now; results will follow within 1-2 days.

These two case studies trace a contamination chain. Nguyen et al. (2025) claimed min-p improves quality and diversity based on flawed evaluation. Jiang et al. (2025, Artificial Hivemind, NeurIPS Best Paper) took this at face value, tested min-p for diversity, and over-generalized to "decoding-time interventions are fundamentally insufficient." Tan et al. (2026, p-less) proposed a new sampler using the same flawed methodology. Three papers, three venues, three author groups — same evaluation failures recurring because no systematic verification mechanism exists.

### Methodological contribution and originality

The reviewer notes that the individual standards "resemble widely discussed best practices." We agree — and this is precisely the puzzle. If these principles are well-known, why do two oral papers at top venues (ICLR 2025, ICLR 2026) violate all four? The gap between knowing best practices and systematically applying them is the contribution:

1. **Best-of-N as a diagnostic protocol.** The analogy is pass@k (Chen et al., 2021): random subsampling is well-known, but Chen et al.'s contribution was repurposing it as a standardized evaluation protocol answering a specific question (how does code generation scale with attempts?). Best-of-N repurposes hyperparameter subsampling to answer a different specific question: does a claimed advantage survive equalized tuning budgets? Grid search selects hyperparameters for one method; Best-of-N compares methods via performance-vs-budget curves. The revision adds Algorithm 1 (pseudocode) and analysis of statistical properties (variance as a function of N, minimum N for reliable conclusions).

2. **Empirical verification at scale.** Applying these principles overturns the central claims of two oral papers at top venues. These violations only surfaced after >6,000 A100-hours of reproduction — the community currently has no systematic mechanism for this kind of post-publication verification.

3. **Operationalized checklist.** All four standards are mapped to concrete pass/fail items, validated against both case studies (details in our response to Reviewer 2LLS).

### Tone and framing

The revision restructures the paper: blueprint and Best-of-N protocol come first as general tools, case studies follow as illustrations. Detailed numerical discrepancies and author interactions move to the appendix.

### Q1: Fair hyperparameter ranges for heterogeneous methods

Best-of-N sidesteps the range-fairness problem by equalizing the *configuration budget* N, not the parameter space. Each method draws N configurations from whatever parameter space is natural for it — no decisions about "equivalent" ranges are needed.

Concretely, from the p-less case study: p-less has 1 hyperparameter (temperature), min-p has 2 (temperature x min-p value). At budget N=20, p-less draws 20 temperature configurations; min-p draws 20 (temperature, p-value) pairs. The single-hyperparameter method covers its space more densely at a given N. This asymmetry is real but *conservative* — it favors the method with more hyperparameters, which has a richer space to exploit. When Best-of-N shows that temperature-only baselines match min-p at equal N, the conclusion strengthens: min-p had the advantage of a richer parameter space and still did not outperform.

If a method's advantage is genuinely that it requires less tuning (as p-less claims), this appears directly in the Best-of-N curve: its performance rises faster at low N. The protocol makes tuning sensitivity visible rather than assuming it away.

### Q2: Venue fit

Evaluation methodology papers are regularly accepted at top ML venues. Three direct precedents: Dodge et al. (2019, "Show Your Work: Improved Reporting of Experimental Results," EMNLP) proposed controlled reporting of hyperparameter search budget — accepted as a main conference paper with no new model or algorithm. Dehghani et al. (2021, "The Benchmark Lottery," NeurIPS) demonstrated that benchmark selection biases rankings — again, no new model. Henderson et al. (2018, "Deep Reinforcement Learning that Matters," AAAI) showed that RL results are fragile under hyperparameter and implementation variation. Our paper fits this lineage and goes further: a formalized diagnostic protocol with pseudocode, >6,000 A100-hours of new experiments across 28 models, corrected re-analyses overturning a high-visibility ICLR Oral, a second case study demonstrating generality, and an operationalized checklist.

The institutional argument is also relevant: if post-publication empirical verification is excluded from main research tracks, the field provides no venue — and therefore no incentive — for this work. The documented cost is that evaluation failures in high-visibility papers propagate uncorrected across three papers at three venues.

---

## Summary of Planned Revisions

1. **Second case study (p-less):** ~1 page demonstrating blueprint generality with new Best-of-N experiments across 28 models
2. **Algorithm 1:** Formal pseudocode for the Best-of-N protocol, with pass@k analogy and analysis of statistical properties (variance as a function of N, minimum N for reliable conclusions)
3. **Grid search distinction:** Explicit paragraph differentiating Best-of-N (diagnostic evaluation protocol producing comparative curves) from grid search (optimization algorithm selecting one configuration)
4. **Operationalized checklist:** Table mapping each standard to concrete items, failure modes, and violations found in both case studies
5. **Related Work section:** Positioning relative to Dodge et al. (2019, EMNLP), Bouthillier et al. (2021), Henderson et al. (2018, AAAI), Dehghani et al. (2021, NeurIPS), Dror et al. (2018), Melis et al. (2020)
6. **Background section:** Motivation, mechanism, and claimed merits of min-p for reader accessibility
7. **Narrative restructuring:** Blueprint and protocol presented first as general tools; case studies as illustrations; adversarial language softened; numerical discrepancies and author interactions moved to appendix
8. **Expanded Limitations section:** Adoption challenges, structural incentives, compute asymmetry between authors and verifiers
9. **Presentation fixes:** URL formatting, Figure 9 sizing, anonymized links, reference format consistency, reduced repetition
10. **Wilcoxon robustness checks:** Non-parametric confirmation of all statistical conclusions
