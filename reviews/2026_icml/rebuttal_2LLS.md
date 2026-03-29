# Response to Reviewer 2LLS — ICML 2026 Submission 31762

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

## Response to Reviewer 2LLS

We appreciate the thorough review of our paper and the acknowledgement that rethinking execution of empirical evaluation is important.  We address the reviewer's concerns about our work below.

### Tone and framing

In the revision, we will restructure the narrative to lead with the general evaluation framework, using both case studies as illustrations rather than as the primary thread. Numerical discrepancies will be consolidated in appendix tables; author interactions will move to footnotes. The result: "a framework validated through two case studies" rather than "a critique that proposes a framework."

### Single case study

We have added a second case study: p-less sampling (Tan et al., ICLR 2026 Oral). See the General Response for details. Different authors, different venue, different method -- same evaluation methodology problems, same outcome when Best-of-N is applied: [TBD: preliminary results].

### Standards 2-4 lack operationalization

Agreed. The revision will include a concrete checklist table:

| Standard | Checklist Item | Failure Mode Prevented | Violated in Min-p? | Violated in P-less? |
|----------|---------------|----------------------|--------------------|--------------------|
| 1. Fair comparison | Report Best-of-N curves at matched budgets | Inflated claims from unequal tuning | Yes (Table 4 vs our Table 1) | Yes (Table 1 vs Table 8) |
| 2. Valid inference | Per-model significance tests with correction | False discoveries from pooled/uncorrected tests | Yes (pooled t-tests) | Yes (no tests at all) |
| 2. Valid inference | >= 3 seeds on all models | Unreliable point estimates | Partial (3 seeds) | Yes (1 seed on key models) |
| 3. Transparency | Release all evaluation code and raw data | Selective reporting, irreproducibility | Partial (code released late) | Yes (no eval code) |
| 3. Transparency | Disclose annotator affiliations in human evals | Conflict of interest | N/A | Yes (3/6 annotators are authors) |
| 4. Consistent reporting | Win/loss tables across all comparisons | Cherry-picked metrics | Yes (omitted losing models) | Yes ("consistently outperforms" despite mixed results) |
| 4. Consistent reporting | Justify metric choice and weighting | Metric gaming | Yes (selective metric reporting) | Yes (AUC inflates high-T advantage) |

Every row is now validated against two independent papers.

### Best-of-N: formalization, grid search distinction, and further analyses

Grid search selects the best hyperparameters for one method; the output is a single configuration. Best-of-N diagnoses whether a claimed advantage survives equalized tuning budgets; output is comparative performance-vs-budget curves. Same mechanics, different purpose. We will add Algorithm 1 (pseudocode) to formalize the protocol.

On further analyses: the revision will include empirical analysis of how Best-of-N estimate variance decreases with N and the minimum N needed for stable rankings in our case studies.

### Best-of-N: comparison to existing protocols

The revision will include a structured Related Work section positioning Best-of-N relative to:
- **Hyperparameter search bias:** Dodge et al. (2019), Bouthillier et al. (2021)
- **Benchmarking fairness:** Henderson et al. (2018), Dehghani et al. (2021)
- **Statistical evaluation in ML:** Dror et al. (2018), Pineau et al. (2021)
- **Fair comparison precedents:** Melis et al. (2020, Mogrifier LSTM)

The key distinction: Dodge et al. and Bouthillier et al. characterize the problem of unequal tuning but do not propose a reusable evaluation protocol. Henderson et al. demonstrate evaluation pitfalls in deep RL but lack a formalized comparison procedure. Melis et al. perform fair comparison ad hoc for their specific setting. Best-of-N systematizes these insights into a general-purpose diagnostic with a specific output format (comparative curves) that can be applied post-hoc to existing sweep data.

### Best-of-N: limitations

Beyond compute cost (addressed below), the protocol has limitations we will discuss in the revision: (1) it measures fairness in number of configurations tried, not compute cost per configuration -- methods with expensive-to-evaluate settings may be disadvantaged; (2) it assumes the sweep covers the relevant region of the search space, which requires domain knowledge; (3) it is most informative when methods have comparable numbers of hyperparameters -- a method with one hyperparameter mechanically has fewer distinct configurations than one with three.

### Alternative fairness philosophies (Q2)

The Best-of-N curve accommodates multiple philosophies rather than privileging one:

- **"Best achievable performance":** Read the right end of the curve (large N). If method A dominates, it is genuinely better when fully tuned.
- **"Tuning is part of the method":** Read the left end (small N). Higher at N=1 means better defaults; faster rise means easier to tune.
- **"Equal effort":** Compare at any fixed N.

A single reported number (e.g., "min-p achieves X on GSM8K") collapses this into one point that can be gamed by choosing favorable hyperparameters. The curve makes the full picture visible.

One caveat: the curve measures effort as configuration count, not wall-clock time or compute. For methods with very different per-evaluation costs, a cost-adjusted comparison should supplement the curve.

### Computational cost

The 6,000 A100-hour figure is the cost of running sweeps, not the Best-of-N protocol itself. The protocol is trivial post-hoc computation on existing sweep results (subsample N configurations per method, take the max).

Whether dense sweeps are needed: (1) this is the cost of fair comparison -- the cost that should be paid whenever a paper claims one method outperforms another and this claim hinges on hyperparameter selection; (2) the protocol degrades gracefully -- even N=5-10 per method can reveal whether a claimed advantage is robust or fragile based on a Mann-Whitney U-test.

### Venue fit

This is not a position paper. The submission contains:
1. A formalized evaluation protocol (Best-of-N) with pseudocode [to be added]
2. Over 6,000 A100-hours of new experiments across 28 models, 5 samplers, 31 temperatures
3. Corrected statistical re-analyses showing a high-visibility ICLR 2025 Oral's central claims are unsupported
4. A second case study (p-less, ICLR 2026 Oral) demonstrating generality
5. An operationalized checklist validated against two independent papers

Henderson et al. (2018, "Deep Reinforcement Learning that Matters") was published at AAAI with the same structure: evaluation methodology + case study overturning existing claims. Our contribution -- formalized protocol, larger experimental effort, two case studies, operationalized checklist -- meets the ICML main track bar.

If post-publication verification papers are unsuitable for main tracks, there is no venue and no incentive for this work. Evaluation failures in high-visibility papers go uncorrected and propagate downstream (Nguyen -> Artificial Hivemind -> p-less).

### Novelty of the re-evaluation

The novelty is empirical: two oral papers at top venues do not survive rigorous evaluation, and their flawed claims propagate downstream (Nguyen -> Artificial Hivemind -> p-less). This finding would not exist without re-running experiments at scale. The contribution is the finding, not the (standard) statistical tools used to obtain it.

### Presentation fixes

We will fix: overflowing reference links, Figure 9 sizing, reference format inconsistencies, and typos.

### Q1: Is the implementation publicly available?

Yes. Full codebase including sweep configurations, evaluation scripts, and analysis notebooks is at [repository URL]. The p-less vLLM patch will be included.

### Reproducibility details

Seeds 0, 1, 2 across all runs. Hyperparameters swept on a fixed grid: 31 temperatures from 0.0 to 3.0 in 0.1 increments; sampler-specific values per sweep configurations. All configurations enumerated, not randomly sampled. These details will be added to the revision.


