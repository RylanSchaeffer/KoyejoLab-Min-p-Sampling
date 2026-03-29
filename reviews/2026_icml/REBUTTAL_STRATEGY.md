# Rebuttal Strategy: ICML 2026 Submission 31762

**Current scores:** Accept (5), Weak Reject (3), Weak Reject (3), Reject (2)
**Target:** Flip all three negative reviewers to at least Weak Accept (4).

---

## Synthesizing Reviewer Objections

### 1. Single Case Study (2LLS, dH2p, cBMY) — CRITICAL

The blueprint claims generality but is demonstrated on one paper. All three negative reviewers raise this. They want us to *apply* Best-of-N to another domain, not just cite others' failures.

### 2. The Four Standards Are Known Best Practices (2LLS, dH2p, cBMY) — CRITICAL

Fair comparison, valid statistics, data transparency, consistent reporting — every reviewer acknowledges these are important but says they are not new. 2LLS: "largely reflect established best practices rather than introducing new methodological tools." cBMY: "do not appear to be genuinely novel, but rather reflect broadly accepted good practices." dH2p: "resemble widely discussed best practices in empirical ML."

### 3. Best-of-N Resembles Grid Search (2LLS, cBMY, KrXT) — CRITICAL

cBMY Q2: "How does the proposed protocol differ from variations of grid search?" 2LLS: "conceptually simple and closely related to standard hyperparameter sweep analyses that examine performance as a function of tuning effort." KrXT Q2 (friendly version): "Earlier work (e.g. Mogrifier LSTM, 2020) have been able to show similar results without the use of it" — i.e., if prior work achieved fair comparison without formalizing a protocol, what does Best-of-N add?

The distinction we failed to make clear: grid search selects hyperparameters for one algorithm; Best-of-N compares algorithms by producing diagnostic performance-vs-budget curves. But this distinction is not in the paper.

### 4. Standards 2-4 Lack Operationalization (2LLS, cBMY) — HIGH

2LLS: "the paper does not propose concrete procedures, frameworks, or artifacts like workflows, checklists, or reporting templates that would operationalize these standards." The four standards are stated as principles but the paper provides no tools to implement them beyond Standard 1 (Best-of-N).

### 5. Best-of-N Protocol Lacks Formalization (2LLS) — HIGH

2LLS: "the protocol lacks proper formalization and further analyses." There is no algorithm box, no pseudocode, no formal definition. The protocol is described in prose within the case study rather than presented as a standalone, reusable method.

### 6. Adversarial Tone / Reads as a Critique (2LLS, dH2p, cBMY) — HIGH

Paper reads as a targeted takedown, not a general methodology. Author interactions and GitHub links feel adversarial. No explanation of what min-p actually is — assumes reader familiarity (cBMY). Repetitive writing, inline links hurt readability, URL overflows, oversized Figure 9.

No reviewer disputes our findings. This is purely framing and presentation.

### 7. Venue Fit — "This Is a Position Paper" (2LLS, dH2p) — HIGH

2LLS: "more naturally suited to venues that emphasize reproducibility, benchmarking, or methodological position papers." dH2p: "reads closer to a position/perspective paper." This is a meta-objection the AC could use to reject even if content concerns are addressed.

### 8. Computational Cost of Best-of-N (2LLS, cBMY) — MEDIUM

cBMY quotes the 6,000 A100-hour figure and questions feasibility. Two layers:
- **Misunderstanding:** The 6,000 hours was the sweep cost, not the protocol cost. The Best-of-N analysis is trivial post-hoc computation.
- **Legitimate concern:** The protocol requires dense sweeps to exist. This is the cost of fair comparison that should have been paid all along, and the protocol degrades gracefully to small sweeps (N=5-10).

### 9. Alternative Fairness Philosophies (2LLS) — MEDIUM

2LLS Q2: "other evaluation philosophies exist (e.g., comparing methods at their best achievable performance, considering tuning as part of the method)." Is equal-effort the right notion of fairness? The Best-of-N curve subsumes multiple philosophies by showing performance at every budget level. If a method's advantage is ease of tuning, its curve rises faster at low N.

### 10. Ethics Flag (cBMY) — MEDIUM

Two concerns: (a) paper reads as public rebuttal implying misconduct, (b) GitHub links de-anonymize authors. Reviewer explicitly hedges ("I do not hold strong opinions"). Fix: anonymize links, state clearly the paper does not allege misconduct.

### 11. Smaller Points

- **Missing Related Work section (2LLS):** "A comparison to existing evaluation frameworks and benchmarking protocols... is missing."
- **Reproducibility gaps (2LLS):** Seed handling and hyperparameter sampling not fully specified.
- **Structural incentives (cBMY Q4):** How can these standards be enforced? Paper doesn't address root causes.
- **Circular significance (cBMY):** Paper's significance depends on min-p's significance.

---

## Prioritized Rebuttal Plan

### 1. Add a Second Case Study — HIGH EFFORT, CRITICAL IMPACT

**Targets:** Objections 1, 7 | **Moves:** 2LLS, dH2p, cBMY

Apply Best-of-N to a paper outside LLM sampling where public sweep data already exists (candidates: papers cited in our intro from Maini et al. 2025, Chandak et al. 2025, or an RL paper). Need not be exhaustive — focused demo on 2-3 models suffices. Kills the single-case-study objection and weakens the position-paper concern.

### 2. Formalize Best-of-N: Algorithm Box + Distinguish from Grid Search — MEDIUM EFFORT, HIGH IMPACT

**Targets:** Objections 3, 5 | **Moves:** 2LLS, dH2p, cBMY, KrXT

- Add Algorithm 1 pseudocode. Makes the contribution tangible and citable.
- Distinguish from grid search: grid search selects hyperparameters for one algorithm; Best-of-N compares algorithms via diagnostic performance-vs-budget curves. Grid search outputs a single configuration; Best-of-N outputs a curve revealing how relative performance changes with budget.
- Compare to prior art: Dodge et al. 2019, Bouthillier et al. 2021, Henderson et al. 2018. Acknowledge Mogrifier LSTM (KrXT Q2) did fair comparison ad hoc; Best-of-N formalizes this into a reusable protocol with a standard diagnostic output.

### 3. Address Computational Cost — LOW EFFORT, MEDIUM IMPACT

**Targets:** Objection 8 | **Moves:** 2LLS, cBMY

Clarify: the protocol is trivial post-hoc analysis; the 6,000 hours was the sweep cost. Acknowledge honestly that sweeps are needed, but argue this is the cost of fair comparison. The protocol degrades gracefully — even N=5-10 per method beats single-config comparisons.

### 4. Address Alternative Fairness Philosophies — LOW EFFORT, MEDIUM IMPACT

**Targets:** Objection 9 | **Moves:** 2LLS

The Best-of-N curve subsumes multiple fairness notions: it shows performance at every budget level N. If a method's advantage is ease of tuning, the curve shows this (steeper rise at low N). If the claim is algorithmic superiority, the curves should separate at high N.

### 5. Operationalize Standards 2-4: Checklist Table — LOW EFFORT, MEDIUM IMPACT

**Targets:** Objection 4 | **Moves:** 2LLS, cBMY

Add a table: each standard → concrete checklist item → failure mode it prevents → where violated in case study. Add statistical test decision tree for Standard 2. This transforms the blueprint from commentary into a tool.

### 6. Restructure Framing + Related Work + Min-p Background — MEDIUM EFFORT, HIGH IMPACT

**Targets:** Objections 6, 7, 11 (missing related work, reproducibility) | **Moves:** 2LLS, dH2p, cBMY

- Rewrite intro: blueprint first, case study second.
- Add "Background: Min-p Sampling" paragraph.
- Add structured Related Work section (Dodge 2019, Bouthillier 2021, Henderson 2018, Dehghani 2021, Dror 2018, Pineau 2021).
- Soften adversarial language. Move author interactions and GitHub links to footnotes/appendix.
- Move numerical discrepancies (7.80 vs 5.80, etc.) to appendix.
- Add implementation details (seeds, sampling) for reproducibility.

### 7. Venue-Fit Response in Rebuttal — LOW EFFORT, HIGH IMPACT

**Targets:** Objection 7 | **Moves:** 2LLS, dH2p

Enumerate: (1) formalized protocol with pseudocode, (2) 6,000+ A100-hours of experiments, (3) corrected statistical re-analysis, (4) second case study, (5) operationalized checklist. Cite Henderson et al. 2018 (AAAI) as precedent.

### 8. Statistical Test Justification — LOW EFFORT, MEDIUM IMPACT

**Targets:** KrXT Q1 | **Moves:** KrXT (retain Accept)

Paired t-tests because paired observations. One-sided to match directional claim. Bonferroni as most conservative correction. IUT because "consistent superiority" = its alternative hypothesis. Wilcoxon yields identical conclusions.

### 9. Ethics Response — LOW EFFORT, MEDIUM IMPACT

**Targets:** Objection 10 | **Moves:** cBMY

Anonymize all identifying links. State clearly: paper does not allege misconduct. Cite Ioannidis 2005 and NeurIPS D&B track as precedent.

### 10. Incentives Paragraph — LOW EFFORT, LOW IMPACT

**Targets:** cBMY Q4 | **Moves:** cBMY

Acknowledge as open problem. Position checklist as first step. Cite ML Reproducibility Checklist adoption as precedent.

### 11. Presentation Fixes — LOW EFFORT, LOW IMPACT

Fix: URL overflows, Figure 9 sizing, "resaerch" typo, repetitive restatements, inline links → footnotes, reference formatting.

---

## Reviewer-by-Reviewer Strategy

| Reviewer | Score | Key Lever | Lead With |
|----------|-------|-----------|-----------|
| KrXT | 5 → retain | Answer Q1 (stat tests) + Q2 (Mogrifier) | Algorithm 1 + Wilcoxon robustness checks |
| 2LLS | 3 → 4+ | Most movable; methodical objections | Second case study + Related Work + fairness answer |
| dH2p | 3 → 4+ | Significance=4 already; needs originality | Algorithm 1 + grid search distinction + venue-fit |
| cBMY | 2 → 3+ | Hardest to move; ethics + originality=1 | Ethics response, then second case study + checklist |

---

## Discussion Notes

### On Objection 1 (Single Case Study): Is a page-limit argument persuasive?

**Considered argument:** "There's no way to do this level of deep analysis on multiple papers AND tell a coherent story AND fit it into 8 pages."

**Verdict: Weak.** Three problems:
1. It implicitly concedes the point — you're agreeing more case studies would be better.
2. Reviewers didn't ask for a second deep analysis. Even a 1-page focused demonstration using existing public sweep data from another domain would satisfy them.
3. "We couldn't fit it" invites: "then the scope doesn't fit this venue."

**Better move:** Actually add a lightweight second case study. Trim adversarial-tone material the reviewers already want cut (author interactions, numerical discrepancies, GitHub links) to reclaim space.

### On Objection 3 (Best-of-N ≈ Grid Search): The real distinction

**Grid search is an optimization algorithm.** Goal: pick the best hyperparameters for your method. Output: one configuration.

**Best-of-N is an evaluation protocol.** Goal: diagnose whether a method's advantage is real or an artifact of unequal tuning. Output: comparative performance-vs-budget curves across methods.

Same underlying mechanics, completely different purpose. The paper never makes this distinction explicit.

**Key analogy:** "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" (Yang et al., arXiv:2504.13837). pass@k is a well-known metric from code generation. Their contribution was not pass@k — it was asking a new question ("does RL teach new reasoning or just improve sampling efficiency?") and showing that pass@k at high k, applied systematically at scale, answers it. They write one sentence: "we extend the commonly used pass@k metric from code generation to all tasks with verifiable rewards." The novelty is the question + the finding, not the metric.

**Our parallel:** Best-of-N subsampling from a sweep is well-known. Our contribution is not the subsampling — it's repurposing it as an evaluation protocol that answers "does this method genuinely outperform, or does the advantage vanish when you equalize optimization budget?" The novelty is the evaluation protocol + the finding. This framing should go in the paper explicitly, in one clear sentence.

### On Objection 10 (Ethics Flag): AC Intervention Already Filed

On 2026-03-26, we filed a confidential AC comment raising two concerns about Reviewer cBMY:

1. **De-anonymization:** cBMY followed a GitHub link, identified an author, and published that author's name in their review — visible to all reviewers. ICML reviewer instructions prohibit this. We requested cBMY's review be removed, a replacement reviewer assigned, and remaining reviewers advised to disregard identifying information.

2. **Ethics flag on scientific critique:** If scrutinizing published work and presenting evidence of its flaws is an ethics violation, misconduct can never be reported. We asked the flag be evaluated on its merits.

No AC response yet as of 2026-03-28.

### The Contamination Chain: Why This Paper Matters Urgently

The absence of rigorous evaluation standards is not a static problem — it compounds. We can now trace a specific causal chain across three papers:

1. **Nguyen et al. (ICLR 2025 Oral)** claims min-p increases both quality and diversity, based on flawed evaluation (unequal tuning, pooled statistics, omitted data).
2. **Artificial Hivemind (NeurIPS 2025 Best Paper)** takes Nguyen et al.'s claim at face value. They test min-p for diversity, get a null result, and over-generalize: "decoding-time interventions are fundamentally insufficient → solutions must come at the training level." But the premise was wrong. Min-p was never genuinely shown to increase diversity. The correct conclusion is not "decoding-time interventions fail" but "min-p was never a proper representative of what decoding-time interventions can do." A flawed upstream claim produced a flawed downstream inference.
3. **P-less Sampling (ICLR 2026 Oral)** proposes yet another sampler and evaluates it using the same methodology our blueprint identifies as flawed: default-hyperparameter baselines, no significance tests, an AUC metric that inflates high-temperature performance. The cycle repeats.

**The argument for the Discussion section (not the rebuttal):** Don't mention our rejection history. Instead:

> Since our initial investigation, the pattern we identified has continued. [Artificial Hivemind] independently found that min-p fails to address output homogeneity, but drew the broader conclusion that decoding-time interventions are fundamentally insufficient — a conclusion that rests on the unverified assumption that min-p was a proper representative of decoding-time methods. Meanwhile, [p-less] proposes a new sampler using the same evaluation methodology our blueprint identifies as flawed: default-hyperparameter baselines, no significance tests, and a metric that inflates high-temperature performance. When we apply the Best-of-N protocol to [p-less]'s evaluation setting, the claimed advantage [shrinks/vanishes]. This illustrates the cost of not adopting rigorous evaluation standards: the field cycles through proposed methods without the tools to distinguish genuine advances from artifacts of unequal comparison.

This says "the field is failing to progress" via concrete evidence, without saying "because our paper was rejected."

### P-less as the Second Case Study: Implementation Path

**Correcting a misconception:** P-less claims to be "hyperparameter-free" but temperature is still a hyperparameter, which they sweep from 0.5 to 2.0. So p-less has 1 hyperparameter (temperature) while other samplers have 2 (temperature + sampler-specific value). This means p-less fits naturally into the Best-of-N framework: at each budget N, p-less draws from its temperature configurations while other samplers draw from temperature × sampler-value configurations. The question becomes: does p-less outperform other samplers when each gets N total configurations?

This is actually a *better* framing than "flat curve" — it tests p-less on its own terms and asks whether the claimed advantage holds under controlled comparison.

**Empirical validation:**
- vLLM v0.7.3 does not support p-less natively. Implementation is ~50 lines (threshold = `probs.square().sum()`).
- We share Mistral-7B-Instruct-v0.1 and GPQA + GSM8K benchmarks with the p-less paper.
- Recommended path: patch vLLM to add p-less as a native sampler, run sweeps on overlapping setup, apply Best-of-N analysis.
- P-less sweeps are smaller than other samplers (only temperature × seeds, no sampler-value dimension).

**Framing:** Not a takedown of p-less. The theoretical contribution (entropy-based adaptive threshold) is genuinely novel. The critique is narrowly about evaluation methodology — the same pattern our blueprint identifies.

### Artificial Hivemind as Cited Evidence

Not a case study for Best-of-N (they don't compare samplers under equal tuning). Instead, cite as independent corroboration:
- NeurIPS 2025 Best Paper directly tests min-p for diversity with aggressive settings (p=0.1, T=2.0), finds 61% of response pairs still exceed 0.8 similarity.
- Strengthens our scientific conclusion that min-p does not deliver on its diversity claims.
- Also serves the contamination chain argument above.

---

## Manuscript Changes Summary

| # | Change | Effort | Impact |
|---|--------|--------|--------|
| 1 | Second case study | High | Critical |
| 2 | Algorithm 1 + grid search distinction | Medium | High |
| 3 | Restructure intro + Related Work + min-p background + implementation details | Medium | High |
| 4 | Venue-fit argument in rebuttal | Low | High |
| 5 | Standards 2-4 checklist table | Low | Medium |
| 6 | Cost clarification paragraph | Low | Medium |
| 7 | Fairness philosophies paragraph | Low | Medium |
| 8 | Statistical test justification + Wilcoxon | Low | Medium |
| 9 | Anonymize links + ethics response | Low | Medium |
| 10 | Contamination chain argument in Discussion (Nguyen → Hivemind → p-less) | Low | High |
| 11 | Cite Artificial Hivemind as independent corroboration of min-p diversity findings | Low | Medium |
| 12 | Incentives paragraph in Discussion | Low | Low |
| 13 | Presentation fixes | Low | Low |
