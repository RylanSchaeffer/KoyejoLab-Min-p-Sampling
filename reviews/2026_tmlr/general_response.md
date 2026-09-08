# General Response (TMLR Submission 11062)

We thank the reviewers, all of whom find the empirical claims supported. Changes:

1. **Why re-examine min-p (WbvC).** The introduction now gives the case: an ICLR 2025 Oral, over 170 citations in one year, and integration into Transformers, vLLM, SGLang, llama.cpp and TGI. Artificial Hivemind (NeurIPS 2025 Best Paper) generalized from min-p's failure to all decoding-time interventions; p-less (ICLR 2026 Oral) used min-p as its primary baseline with the same evaluation methodology.

2. **Tone (WbvC, wjeg, fy93).** We removed the two sentences wjeg flagged, cut commentary on the original authors and the ICLR review process, shortened block quotations, and restated criticisms as what the data show. A new Sec. 6 paragraph separates field-wide practices from issues specific to this paper.

3. **Human evaluation result (wjeg).** Sec. 6 no longer calls the single Bonferroni-surviving comparison a "benefit." It names it (quality, min-p vs. basic, T=3.0), notes every sampler scores lower at T=3.0 than at standard temperatures, and does not read it as evidence min-p improves quality or diversity. The abstract and Sec. 2.4 are unchanged; Table 1's caption now names the comparison.

4. **LLM-as-a-judge (wjeg).** Sec. 4.1 now separates the indirect-design and non-transitivity arguments, and states that Xu et al. (2025) postdates the ICLR 2025 review cycle.

5. **Benchmark scope (wjeg, fy93).** Sec. 3 now adds GPQA with the identical grid (four samplers, 31 temperatures, six values per sampler, three seeds) on 16 of 18 models. The result matches GSM8K CoT: under equal budgets, the best min-p configuration is within about one point of the best other sampler on every model, above on two, below on three, indistinguishable on the rest. The abstract, Sec. 3 and limitations now say both original NLP benchmarks are covered. An appendix figure reports GSM8K CoT on ten larger models (Qwen 2.5 14B, 32B, 72B; Gemma 2 27B; Llama 3.1 70B; base and instruct), same conclusion.

6. **"Lightly edited" hyperparameters (wjeg).** Sec. 3.1 now lists, per sampler, the original values, ours, and why each changed.

7. **Minor.** Ackley et al. (1985) is justified in a footnote. The Table 1 typo and quotation marks are fixed.

Detailed responses follow under each review.
