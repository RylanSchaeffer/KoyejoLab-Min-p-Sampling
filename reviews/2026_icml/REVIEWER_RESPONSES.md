# Reviewer Responses — ICML 2026 Submission 31762

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

## Response to Reviewer 2LLS

We address each concern below.

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

Grid search selects the best hyperparameters for one method; output is a single configuration. Best-of-N diagnoses whether a claimed advantage survives equalized tuning budgets; output is comparative performance-vs-budget curves. Same mechanics, different purpose. We will add Algorithm 1 (pseudocode) to formalize the protocol.

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

Whether dense sweeps are needed: (1) this is the cost of fair comparison -- the cost that should be paid whenever a paper claims one method outperforms another; (2) the protocol degrades gracefully -- even N=5-10 per method reveals whether a claimed advantage is robust or fragile.

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
