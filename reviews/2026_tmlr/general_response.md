# General Response (TMLR Submission 11062)

We thank all three reviewers for their careful reading. All three reviewers find the empirical claims supported by the evidence. We have revised the manuscript in response to every requested change. The main changes are:

1. **Why min-p warrants re-examination (WbvC).** The introduction now makes the case explicitly. Beyond its ICLR 2025 Oral, the paper accrued more than 170 citations in its first year, min-p was integrated into most widely used open-source inference libraries (Transformers, vLLM, SGLang, llama.cpp, TGI), and its claims shaped subsequent work at top venues: Artificial Hivemind (NeurIPS 2025 Best Paper) generalized from min-p's failure to a conclusion about all decoding-time interventions, and p-less (ICLR 2026 Oral) adopted min-p as its primary baseline and repeated the same evaluation methodology.

2. **Tone (WbvC, wjeg, fy93).** We removed the two sentences wjeg flagged as condescending, cut commentary on the original authors and on the ICLR review process that carried no evidential content, shortened block quotations, and rewrote editorializing language throughout so that criticisms are stated as what the data show. A new paragraph in Sec. 6 separates practices that are common across the field from issues specific to the original paper.

3. **Consistent statement of the human evaluation result (wjeg).** Sec. 6 no longer describes the single Bonferroni-surviving comparison as a "benefit." It now names the comparison (quality, min-p vs. basic, at T=3.0), notes that every sampler including min-p scores lower at that temperature than at standard temperatures, and states that we do not read it as evidence min-p improves quality or diversity. The abstract and Sec. 2.4 are unchanged; Table 1's caption now names the surviving comparison so the three passages visibly agree.

4. **LLM-as-a-judge critique (wjeg).** Sec. 4.1 now presents the indirect-design argument and the non-transitivity argument as separate points, and states that Xu et al. (2025) postdates the ICLR 2025 review cycle, so the original authors could not have been expected to account for it.

5. **Scope of the benchmark evidence (wjeg, fy93).** The abstract, Sec. 3 and the limitations now scope our sweeps to GSM8K CoT. We also added GPQA, the original paper's other NLP benchmark, using the identical grid (four samplers, 31 temperatures, six values per sampler, three seeds) on 16 of the 18 models. The result matches GSM8K CoT: under equal hyperparameter budgets, the best min-p configuration is within about one percentage point of the best other sampler on every model, above on two, below on three, and indistinguishable on the rest. An appendix figure additionally reports GSM8K CoT on ten larger models (Qwen 2.5 14B, 32B and 72B, Gemma 2 27B, Llama 3.1 70B, base and instruct).

6. **"Lightly edited" hyperparameters (wjeg).** Sec. 3.1 now lists, per sampler, the original paper's values, ours, and the reason for each change.

7. **Minor.** Ackley et al. (1985) is now justified in a footnote. The Table 1 typo and the quotation marks are fixed.

Detailed responses follow under each review.
