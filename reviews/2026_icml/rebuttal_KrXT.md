# Response to Reviewer KrXT

## General Response

Three of four reviewers note the blueprint relies on a single case study. We now have a second.

**Second case study: p-less sampling.** We applied Best-of-N to p-less (Tan et al., ICLR 2026 Oral), which claims to "consistently outperform." We implemented p-less in vLLM v0.7.3 and are sweeping 28 models on GSM8K CoT and GPQA (5,952 runs, 3 seeds). Best-of-N results will follow within 1-2 days.

P-less has a novel theoretical contribution (Renyi entropy) and real efficiency gains. Our critique targets evaluation, not the method. Independent of Best-of-N, it violates all four standards: default-only baselines (tuned min-p matches p-less on GPQA per their Table 8); no significance tests on accuracy despite having them for efficiency; human eval at T=2.0 vs T=1.0 with 3/6 author annotators; and "consistently outperforms" claimed when min-p wins 2/4 datasets on Llama3-70B, p-less loses 3/4 at T=1.0, and ranks last in creative writing at T=1.0. Different paper, authors, venue: same problems.

**Independent corroboration.** Jiang et al. (Artificial Hivemind, NeurIPS 2025 Best Paper) tested min-p for diversity across 70+ models: 61% of pairs exceeded 0.8 similarity. They over-generalized to "decoding-time interventions are insufficient": based on the unverified assumption that min-p represents decoding-time methods.

**Revisions.** Blueprint and Best-of-N presented first; case studies as illustrations. Adding Related Work (Dodge 2019, Bouthillier 2021, Henderson 2018, Dehghani 2021, Dror 2018, Melis 2020), Algorithm 1 (pseudocode), and operationalized checklist.

---

## Response to Reviewer KrXT

We thank the reviewer for the careful evaluation.

**Q1: Statistical tests.** Each choice follows from the data and the claim under test. Paired t-tests: each (model, temperature, seed) triple evaluates all samplers on the same prompts; pairing controls for model and prompt variance. One-sided: the claim is directional ("min-p outperforms"). Bonferroni: the most conservative correction: Nguyen et al. claimed dominance across 12 settings; under Bonferroni, only 1 of 12 rejects the null. IUT: "consistent superiority" requires the effect in every setting simultaneously; the null is the union of "min-p does not outperform in setting x"; we fail to reject. The revision includes Wilcoxon signed-rank tests as a non-parametric robustness check.

**Q2: Best-of-N vs Mogrifier LSTM.** Best-of-N detects cherry-picking by equalizing hyperparameter budget. In Nguyen et al., min-p received a more extensive sweep than top-p, making it appear superior. Melis et al. (2020) demonstrated the same insight ad hoc; Best-of-N formalizes it into a reusable protocol with performance-vs-budget curves and pseudocode (Algorithm 1 in revision).

Grid search optimizes hyperparameters for one method (output: one configuration). Best-of-N diagnoses whether a claimed advantage survives equalized budgets (output: comparative curves). Like pass@k (Chen et al., 2021), the contribution is not the mechanism but its application as a diagnostic protocol.

**On "fair comparison through hyperparameter search is the main driving factor."** Exactly right: and that is the finding, not a limitation. Baselines match or exceed min-p at equal budgets. The advantage was an artifact of unequal tuning. That equalization is "the main driving factor" confirms the protocol's value: it identifies the mechanism by which claims were inflated. The p-less case study shows the same pattern from different authors at a different venue.
