# Rebuttal Strategy: TMLR Submission 11062

**Current assessments (TMLR's two acceptance criteria):**

| Reviewer | Claims supported? | Audience interested? |
|----------|-------------------|----------------------|
| wjeg     | Yes               | Yes                  |
| WbvC     | Yes               | **No**               |
| fy93     | Yes               | Yes                  |

**Target:** Flip WbvC's "No" on audience interest, and satisfy wjeg's two Critical requested changes. No reviewer disputes the empirical findings.

**Action Editor:** Matt Kusner

---

## Synthesizing Reviewer Objections

### 1. Why does min-p warrant re-examination? (WbvC) — CRITICAL

WbvC is the only reviewer answering "No" on a TMLR acceptance criterion. Argument: no frontier lab has adopted min-p; most users never touch samplers; GitHub stars do not establish influence. WbvC explicitly says: "My assessment on this could change if there's evidence of broader adoption that I have overlooked."

wjeg and fy93 independently argue the opposite: "samplers affect every LLM use" and "papers the community has platformed, promoted and set apart as exemplary should hold under scrutiny."

### 2. Adversarial tone (WbvC, wjeg, fy93) — CRITICAL

Raised by all three reviewers, and by three of four ICML reviewers before them.
- WbvC: "too strong and unnecessarily adversarial throughout"; "very uncomfortable to read this paper at times"; objects to "challenging the original authors' claims of community adoption."
- wjeg: "For those unfamiliar with AlpacaEval" (Sec. 4.1) reads as condescending; the library analogy closing Sec. 5.2 "reads as condescending"; the non-transitivity point should "not add negligence" to the indirect-design evidence.
- fy93: "I personally find the title a bit aggressive" (but says it does not change their rating).

### 3. Human-eval conclusion is internally inconsistent (wjeg, Critical) — HIGH

Sec. 2.4 and the abstract say "no apparent advantage." Sec. 6 says the data "weakly suggest" a benefit at higher temperatures. Table 1 has one Bonferroni-surviving result (quality, min-p vs. basic, high temperature).

### 4. LLM-as-a-judge argument conflates two critiques and has a chronology problem (wjeg, Critical) — HIGH

The indirect-design critique (every sampler compared against basic at fixed temperature, not head-to-head) stands alone. The non-transitivity critique depends on Xu et al. (2025), which was posted February 2025 and accepted to ICML 2025, after the ICLR 2025 review cycle. wjeg wants the two separated and the chronology acknowledged.

### 5. Benchmark evidence is only GSM8K CoT (wjeg, fy93) — MEDIUM

Original paper claimed superiority "across benchmarks"; GPQA is untested in the TMLR manuscript. wjeg wants Sec. 3's conclusion and headline scoped accordingly. fy93 asks about newer models, other benchmarks, and open-ended or MBR-style tasks where diversity could matter (Freitag et al. 2023).

### 6. "Lightly edited" hyperparameters unexplained (wjeg) — MEDIUM

Sec. 3.1 says hyperparameters were taken from the original paper but "lightly edited." Which values changed, and why?

### 7. Dominated-region Pareto critique framed as min-p specific (wjeg, also Broader Impact) — MEDIUM

Reporting wins in a practically suboptimal region of the quality-diversity trade-off is a community-wide practice. wjeg asks us to distinguish community-wide oversights from issues specific to min-p, "operating in good faith throughout."

### 8. Minor presentation (wjeg, fy93) — LOW

- Ackley 1985 citation for standard sampling: justify or replace (fy93).
- Tab. 1: "Bonferroni correcting" -> "Bonferroni correction" (fy93).
- Quotation marks render incorrectly in the PDF, e.g. "Llama" and Sec. 3.1 (wjeg).
- Block quotations take too much space; tighten or move to appendix (wjeg).
- Several claims rest on public exchanges with the original authors that readers cannot verify from the paper (wjeg, Weaknesses).

---

## Prioritized Rebuttal Plan

### 1. Make the Case That Min-p Matters — LOW EFFORT, CRITICAL IMPACT

**Targets:** Objection 1 | **Moves:** WbvC (the only "No")

Do not rely on GitHub stars; WbvC has already discounted them. Build the case on evidence WbvC has not considered:
- Peer-review platforming: ICLR 2025 Oral, 18th highest-scoring submission. The venue itself declared the claims exemplary.
- Integration into the inference stack: Hugging Face Transformers, vLLM, llama.cpp, SGLang, TGI. Every user of these libraries can turn min-p on; several UIs expose it by default.
- Downstream research built on the claims: Artificial Hivemind (Jiang et al., NeurIPS 2025 Best Paper) takes min-p's diversity claims at face value and generalizes to "decoding-time interventions are insufficient"; p-less (Tan et al., ICLR 2026 Oral) adopts min-p as its primary baseline and repeats the same evaluation methodology.
- Reframe the criterion: the value of a re-examination scales with the visibility of the claim, not the market share of the method. TMLR's criterion is whether *some* of its audience would be interested; wjeg and fy93 say yes for reasons independent of adoption.

Add this framing to the introduction, not just the rebuttal. **TODO: verify current inference-library integration list and cite the p-less and Hivemind papers.**

### 2. Tone Pass — MEDIUM EFFORT, CRITICAL IMPACT

**Targets:** Objection 2 | **Moves:** WbvC, wjeg, fy93

Concrete edits:
- Delete "For those unfamiliar with AlpacaEval" (Sec. 4.1) and the library analogy closing Sec. 5.2.
- Move author-exchange narrative, GitHub issue links, and long block quotes to an appendix. Keep only what a reader needs to verify a claim from the paper itself (also addresses wjeg's "cannot verify from the paper" weakness by pointing to the released data instead).
- Remove editorializing adjectives. Every criticism should be a statement of what the data show.
- Decide on the title. fy93 says it does not affect their rating; WbvC did not mention it. **Rylan's call.**
- Do not defend the Sec. 5 adoption critique on tone grounds; instead reframe it as a factual check that the original authors themselves accepted (the numbers were retracted). Compress the section.

Rebuttal language: acknowledge directly, do not argue that the tone was justified.

### 3. Reconcile the Human-Eval Conclusion — LOW EFFORT, HIGH IMPACT

**Targets:** Objection 3 | **Moves:** wjeg (Critical #1)

Adopt one phrasing everywhere: min-p shows no *consistent* advantage; one of twelve comparisons (quality, min-p vs. basic, at the highest temperature) survives Bonferroni correction; the IUT fails to reject. Update abstract, Sec. 2.4, and Sec. 6 to match. The single surviving result should be stated in the abstract, not only in the limitations.

### 4. Separate Indirect-Design from Non-Transitivity; Acknowledge Chronology — LOW EFFORT, HIGH IMPACT

**Targets:** Objection 4 | **Moves:** wjeg (Critical #2)

Restructure Sec. 4.1 into two labeled arguments. State explicitly that Xu et al. (2025) postdates the ICLR 2025 review cycle, so the original authors could not have been expected to account for it; the point is about what the evidence supports today, not about negligence.

### 5. Scope Sec. 3 to GSM8K CoT, or Add GPQA — LOW/MEDIUM EFFORT, MEDIUM IMPACT

**Targets:** Objection 5 | **Moves:** wjeg, fy93

Minimum: change the Sec. 3 headline and abstract to "on GSM8K CoT." Better: add GPQA. The ICML rebuttal sweeps already produced GPQA Best-of-N results across 18 models (5,022 runs); check whether the min-p-vs-baselines GPQA panel can be dropped in with little new compute. **TODO: check existing GPQA sweep coverage for the nine TMLR models.**

For fy93's newer-models and MBR/open-ended asks: acknowledge in limitations. fy93 says it would not change their rating.

### 6. Document the "Lightly Edited" Hyperparameters — LOW EFFORT, MEDIUM IMPACT

**Targets:** Objection 6 | **Moves:** wjeg

Add a table or footnote: original value, our value, reason. **TODO: diff sweep YAMLs against the original paper's Table/Appendix to enumerate exactly what changed.**

### 7. Reframe the Pareto Critique as Community-Wide — LOW EFFORT, MEDIUM IMPACT

**Targets:** Objection 7 | **Moves:** wjeg

One paragraph: reporting wins in a dominated region is common practice; we note it because the original paper's trade-off claim rests on it. Separate "common oversight" from "specific to this paper" (omitted data, mismatched Table 3(b) reporting, unsubstantiated adoption numbers).

### 8. Presentation Fixes — LOW EFFORT, LOW IMPACT

**Targets:** Objection 8

- Ackley 1985: decide whether to keep with justification (Boltzmann machine origin of temperature sampling) or cite a more standard source. **Rylan's call.**
- Fix "Bonferroni correcting" in Tab. 1.
- Fix quotation marks (likely straight `"` instead of ``` `` ``` / `''` in LaTeX).
- Tighten block quotes.

---

## Reviewer-by-Reviewer Strategy

| Reviewer | Status | Key Lever | Lead With |
|----------|--------|-----------|-----------|
| WbvC | Yes / **No** | Only "No"; explicitly open to changing on adoption evidence | Adoption evidence beyond stars (platforming, inference stack, downstream papers), then a direct acknowledgement of tone with concrete edits |
| wjeg | Yes / Yes | Two Critical requested changes; strongly supportive | Fixed human-eval conclusion, split LLM-judge argument with chronology, scoped Sec. 3, hyperparameter table |
| fy93 | Yes / Yes | Minor; supportive | Ackley citation decision, typo fix, limitations note on open-ended/MBR tasks and newer models |

---

## Rylan's Judgements (per issue)

### 1. Min-p significance case — DECIDED

Adopt the visibility-and-consequences framing. Add it to the introduction and the rebuttal. Evidence, in order of strength:

- **Citations:** ~171 citations within a year of publication (Google Scholar, Sept 2026; **TODO: verify exact count and date before submitting**).
- **Peer-review platforming:** ICLR 2025 Oral, 18th highest-scoring submission.
- **Inference-stack integration:** Hugging Face Transformers, vLLM, llama.cpp, SGLang, TGI. The integrations were in part obtained on the strength of the paper's claims, and the integrations were then cited back as evidence of adoption (the Camera Ready's revised adoption statement). Say this carefully: the point is that the paper's credibility and the library integrations reinforced each other, not that anyone acted in bad faith.
- **Downstream research contaminated by the claims:**
  - *Artificial Hivemind* (Jiang et al., NeurIPS 2025 Best Paper) tested min-p as its representative decoding-time intervention (p=0.1, T=2.0), found mode collapse persisted (61% of response pairs above 0.8 similarity), and concluded that "more generalizable solutions are needed at the model training level." That generalization from one sampler to all decoding-time interventions rests on min-p being a strong diversity method, which it is not. Context: `reviews/2026_icml/artificial_hivemind.md`. **Caution:** Hivemind also writes "min-p is not widely adopted." Do not quote that section; WbvC could cite it back.
  - *p-less sampling* (Tan et al., ICLR 2026 Oral) adopts min-p as its primary baseline and repeats the same evaluation failures: default-only baselines, no significance tests on accuracy, human eval at mismatched temperatures with author annotators, "consistently outperforms" contradicted by its own Table 1. Context: `reviews/2026_icml/p_less_sampling.md`.
- **Reframe the criterion:** a re-examination's value tracks the prominence of the claim, not the market share of the method. TMLR asks whether *some* of its audience would be interested; wjeg and fy93 say yes for reasons independent of adoption ("samplers affect every LLM use").

### 2. Tone / title — DECIDED

- Cut the two flagged sentences ("For those unfamiliar, AlpacaEval reports win rates..." in `04_llm_as_judge_evals.tex:20`; "akin to publishing a book and then claiming credit for the library" in `05_community_adoption.tex:46`). Tell reviewers explicitly that both are removed.
- Send a background agent through all TMLR `.tex` files to sand down the harshest edges: editorializing adjectives, rhetorical flourishes, sentences about the authors rather than the evidence. Report the edits as a diff for Rylan to approve.
- Title: unchanged for now (fy93 says it does not affect their rating; WbvC did not raise it).

### 3. Human-eval conclusion phrasing — NEEDS A COMPACT FORMULATION

The finding: min-p's only statistically detectable advantage occurs in a regime (high temperature) where every sampler, including min-p, scores worse in absolute terms than at lower temperatures. In the regimes that achieve the best quality and diversity, min-p is indistinguishable from baselines. The problem is that this is too long to state everywhere.

Proposed compact phrasing, using "Pareto-dominated regime" as the carrier concept (Sec. 2.4 already introduces the trade-off framing, so the term is earned):

- **Abstract (one clause):** "our reanalysis demonstrates min-p did not outperform baselines in quality, diversity, or a trade-off between quality and diversity, except in a high-temperature regime where all samplers, including min-p, perform worse."
- **Sec. 2.4 bold conclusion:** "Min-p's only advantage appears in a Pareto-dominated regime; at the temperatures that achieve the best quality and diversity, min-p is indistinguishable from basic and top-p."
- **Sec. 6:** keep the current "weakly suggest" sentence, but make it echo the same phrase: "The data weakly suggest min-p can help at higher temperatures, but that regime is Pareto-dominated: every sampler, including min-p, achieves higher quality and diversity at lower temperatures."
- **Table 1 caption / text:** state explicitly which comparison survives Bonferroni (quality, min-p vs. basic, at the highest temperature) so the reader can connect it to the regime argument.

The three statements then say the same thing at three lengths, which is what wjeg asked for.

### 4. LLM-judge split + chronology — DECIDED (see explanation in chat)

Sec. 4.1 currently makes two arguments in one breath:

- **(a) Indirect design.** Every sampler was compared against basic at T=1.0, so min-p was never tested head-to-head against top-p. This argument stands on its own and does not depend on any citation.
- **(b) Non-transitivity.** Even if A beats C and B beats C, one cannot infer A beats B, because LLM-judge preferences are not transitive (Xu et al. 2025).

The manuscript writes "The authors' design choice is additionally concerning because LLM-judge preferences are probably not transitive, as shown by recent research." wjeg's objection: Xu et al. was posted Feb 2025 and accepted at ICML 2025, after the ICLR 2025 review cycle closed, so phrasing (b) as a further fault of the authors' *choice* implies they should have known something that did not yet exist. wjeg agrees the critique is valid; they want it stated as "what the evidence supports today" rather than as negligence.

Fix: two short labeled paragraphs. State (a) first as the design critique. Then state (b) as an independent inferential point: "Separately, subsequent work posted after the ICLR 2025 review cycle (Xu et al., 2025) shows LLM-judge preferences are not transitive, so indirect comparisons of this kind cannot in general be chained into head-to-head conclusions." Drop "additionally concerning."

### 5. GSM8K scoping vs. adding GPQA — PENDING
### 6. "Lightly edited" hyperparameters — PENDING
### 7. Pareto critique reframing — PENDING
### 8. Ackley citation / minor fixes — PENDING
