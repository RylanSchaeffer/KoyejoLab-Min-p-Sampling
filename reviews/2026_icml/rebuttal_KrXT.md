# Response to Reviewer KrXT

## General Response

Three of four reviewers note that our blueprint relies on a single case study. We have since added a second: p-less sampling (Tan et al., ICLR 2026 Oral), which claims to "consistently outperform existing sampling approaches." We implemented p-less in vLLM v0.7.3 and are running sweeps across 28 models (5,952 runs, 3 seeds each). Best-of-N results will follow within 1-2 days. We want to be clear: p-less has a real theoretical contribution (connecting thresholds to Renyi entropy) and genuine efficiency gains. Our critique targets the evaluation methodology, not the method itself.

Even without Best-of-N, the p-less paper violates all four of our standards. Baselines use default hyperparameters only (their own Table 8 shows tuned min-p matches p-less on GPQA). Significance tests appear for efficiency but are absent for accuracy. The human evaluation compares T=2.0 vs T=1.0, and 3 of 6 annotators are authors. The paper claims "consistent" superiority, but min-p wins 2/4 datasets on Llama3-70B and p-less loses 3/4 at T=1.0. Different paper, different authors, different venue, same problems.

Separately, Jiang et al. (Artificial Hivemind, NeurIPS 2025 Best Paper) independently tested min-p for diversity across 70+ models. They found 61% of response pairs exceeded 0.8 similarity, corroborating our finding that min-p's diversity claims are unsupported.

The revision will restructure the paper so the blueprint comes first, with case studies as illustrations. We are also adding a Related Work section, Algorithm 1 (Best-of-N pseudocode), and an operationalized checklist validated against both case studies.

---

## Response to Reviewer KrXT

We thank the reviewer for the careful and positive evaluation.

**Q1: Can you elaborate on the selection and justification of statistical tests?**

Each test follows from the data structure and the claim being tested.

We use *paired t-tests* because every sampler sees the same prompts within each (model, temperature, seed) triple. Pairing controls for model and prompt variance. The tests are *one-sided* because the original claim is directional: "min-p outperforms."

We apply *Bonferroni correction* because it is the most conservative standard option; any rejection survives scrutiny. The result: Nguyen et al. claimed min-p dominates across 12 settings, but under Bonferroni correction, only 1 of 12 rejects the null.

We then apply an *Intersection-Union Test* (IUT) because "consistent superiority" requires the effect to hold in every setting simultaneously. Under IUT, the null hypothesis is the union of "min-p does not outperform in setting k." We fail to reject.

The revision will add Wilcoxon signed-rank tests as a non-parametric robustness check.

**Q2: How important is Best-of-N? Prior work (e.g., Mogrifier LSTM) showed similar results without it.**

Best-of-N detects a specific problem: inflated claims caused by unequal hyperparameter budgets. In Nguyen et al., min-p received a more extensive sweep than top-p, making it appear superior. Equalizing the budget eliminates the advantage. Best-of-N is a powerful way to sanity check superiority claims and identify cherry-picking.

Melis et al. (2020) demonstrated this same insight for LSTMs, but their analysis was ad hoc and setting-specific. Best-of-N formalizes the idea into a reusable protocol with performance-vs-budget curves and pseudocode (Algorithm 1 in the revision). It is not always necessary, but when cherry-picking is suspected, it provides definitive evidence.

The key distinction from grid search: grid search is an *optimization* protocol that picks the best hyperparameters for one method. Best-of-N is an *evaluation* protocol that tests whether a claimed advantage survives when all methods receive equal tuning budget.

**On the concern that "fair comparison through hyperparameter search is the main driving factor."**

Yes, and that is the finding, not a limitation. Baselines match or exceed min-p once they receive equal tuning budget. The advantage was an artifact of unequal tuning. The fact that equalization is "the main driving factor" confirms the protocol's value: it identifies exactly how the original claims were inflated. The p-less case study shows the same pattern from different authors at a different venue.
