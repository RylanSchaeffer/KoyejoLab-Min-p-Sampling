# Response to Reviewer KrXT

## General Response

Three of four reviewers note that the blueprint relies on a single case study. Since submission, we have added a second: p-less sampling (Tan et al., ICLR 2026 Oral), a truncation sampler claiming to "consistently outperform existing sampling approaches." We implemented p-less in vLLM v0.7.3 and are running sweeps across 28 models (5,952 runs, 3 seeds each). Best-of-N results will follow within 1-2 days. P-less has a novel theoretical contribution (connecting thresholds to Renyi entropy) and real efficiency gains; our critique targets evaluation methodology, not the method itself.

Independent of Best-of-N, the p-less paper violates all four standards: baselines at default hyperparameters only (their own Table 8 shows tuned min-p matches p-less on GPQA); no significance tests on accuracy despite including them for efficiency; human evaluation comparing T=2.0 vs T=1.0 with 3 of 6 author annotators; and "consistently outperforms" claimed when min-p wins 2/4 datasets on Llama3-70B and p-less loses 3/4 at T=1.0. Different paper, different authors, different venue, same problems.

Separately, Jiang et al. (Artificial Hivemind, NeurIPS 2025 Best Paper) independently tested min-p for diversity across 70+ models and found 61% of response pairs exceeded 0.8 similarity, corroborating our finding that min-p's diversity claims are unsupported.

The revision will restructure the paper so the blueprint comes first, with case studies as illustrations. We are adding a Related Work section, Algorithm 1 (Best-of-N pseudocode), and an operationalized checklist validated against both case studies.

---

## Response to Reviewer KrXT

We thank the reviewer for the careful evaluation.

**Q1: Statistical tests.** Each choice follows from the data and the claim under test. We use paired t-tests because every sampler is evaluated on the same prompts for each (model, temperature, seed) triple, controlling for model and prompt variance. Tests are one-sided because the claim is directional ("min-p outperforms"). We apply Bonferroni correction, the most conservative standard option, so any rejection is robust to criticism. Concretely, Nguyen et al. claimed dominance across 12 settings; under Bonferroni, only 1 of 12 rejects the null. We then apply an Intersection-Union Test because "consistent superiority" requires the effect in every setting simultaneously; the null is the union of "min-p does not outperform in setting x." We fail to reject. The revision will include Wilcoxon signed-rank tests as a non-parametric robustness check.

**Q2: Best-of-N vs Mogrifier LSTM.** Best-of-N detects cherry-picking by equalizing hyperparameter budget across methods. In Nguyen et al., min-p received a more extensive sweep than top-p, making it appear superior. Melis et al. (2020) demonstrated the same insight but their analysis was ad hoc and setting-specific. Best-of-N formalizes this into a reusable protocol with performance-vs-budget curves and pseudocode (Algorithm 1 in the revision). In some domains this is infeasible, but where cherry-picking is suspected, it provides definitive evidence.

The distinction from grid search: grid search is an optimization protocol that picks the best hyperparameters for one method. Best-of-N is an evaluation protocol that tests whether a claimed advantage survives when all methods receive equal tuning budget, producing comparative performance-vs-budget curves.

**On "fair comparison through hyperparameter search is the main driving factor."** Exactly right, and that is the finding, not a limitation. Baselines match or exceed min-p at equal budgets. The advantage was an artifact of unequal tuning. That equalization is "the main driving factor" confirms the protocol's value: it identifies the mechanism by which claims were inflated. The p-less case study shows the same pattern from different authors at a different venue.
