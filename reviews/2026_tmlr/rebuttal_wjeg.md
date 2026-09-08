# Response to Reviewer wjeg

We thank the reviewer. We addressed every requested change.

**Critical 1: Reconcile Sec. 2.4 and the abstract with Sec. 6.** Sec. 6 was wrong to call the one Bonferroni-surviving comparison a "benefit." A win in a regime where every method scores worse is not a benefit anyone would choose, so the abstract and Sec. 2.4 were right not to describe it as one. It now names the comparison (quality, min-p vs. basic, T=3.0), notes that at T=3.0 every sampler, min-p included, scores lower on quality and diversity than at standard temperatures, and states that we do not read it as evidence that min-p improves either. The Table 1 caption names the same comparison.

**Critical 2: Separate indirect design from non-transitivity; acknowledge chronology.** Sec. 4.1 now gives the indirect-design argument alone: every sampler was compared against basic at T=1.0, so min-p was never tested against top-p. Non-transitivity is presented separately. We state that Xu et al. (2025) was posted after the ICLR 2025 review cycle concluded, so the original authors could not have been expected to account for it. We removed "the authors' design choice is additionally concerning."

**"Lightly edited" hyperparameters.** Sec. 3.1 now documents this. Min-p and top-p are strict supersets of the original four values each, plus two toward the extremes: 0.01 and 0.02 for min-p, since the original paper calls small p the sensitive regime; 0.98 and 0.99 for top-p, toward no truncation. Top-k was the one substantive edit: the original {10, 15, 20, 40, 50, 180} has four values between 10 and 50; ours, {10, 30, 50, 100, 150, 200}, keeps 10 and 50 and spaces the rest evenly. Temperatures went from the original {0.7, 1.0, 1.5, 2.0, 3.0} to 0.1 spacing. Every sampler value ran at every temperature with three seeds; the original released logs cover a ragged subset of the grid with one run each. We also corrected the text: the values came from the original paper's text, appendix tables and released evaluation logs, since the main text names only two values per sampler.

**Scope Sec. 3 to GSM8K CoT.** We extended the evidence rather than narrowing the claim. Sec. 3 now includes GPQA, the original paper's other benchmark, with the identical sweep on 16 of the 18 models. The two Gemma 2 Instruct models produced no GPQA scores under the pinned lm-eval version and are omitted; the caption says so. The conclusion matches GSM8K CoT: at equal hyperparameter budgets, the best min-p configuration is within 1.3 percentage points (six of 448 questions) of the best other sampler on every model, numerically higher on 9, equal on 3, lower on 4, with every margin smaller than the seed-to-seed noise of a single configuration. The abstract now reads "comprehensively sweeping the original paper's NLP benchmarks (GSM8K CoT and GPQA)." The Sec. 3 headline and the Sec. 3.1 compute statement cover both benchmarks. Sec. 6 still notes that the sweeps cover only models released through 2024 and accuracy-style benchmarks.

**Dominated-region critique.** Sec. 2.4 now says this practice is common in the sampling literature and that we highlight it only because the original paper's trade-off claim rests on such a region. Per the Broader Impact comment, a new Sec. 6 paragraph, "Which Issues Are Specific to This Paper?", separates community-wide practices (dominated-region reporting, uncorrected multiple comparisons, indirect LLM-as-a-judge designs) from issues specific to the original paper (omitted basic-sampler data, mismatched reporting in Table 3(b), retracted adoption figures).

**"For those unfamiliar with AlpacaEval."** Removed.

**Block quotations.** Reduced from seven to two. Five are now inline quotations of the key phrase. The Area Chair and reviewer quotes moved from Sec. 5 to an appendix. The remaining two are the original and revised adoption statements.

**Quotation marks.** Fixed throughout. The source had straight closing quotes.

**Library analogy in Sec. 5.2.** Removed.

**Claims resting on public exchanges.** Where a claim rests on an exchange with the original authors, the manuscript links to the public record or the resulting artifact (the released data file, the Camera Ready revision). If the reviewer would like any claim supported differently, we will revise it.
