# Response to Reviewer fy93

We thank the reviewer.

**Ackley et al. (1985) for standard sampling.** To our knowledge it is the earliest work to sample discrete states from a temperature-scaled Boltzmann distribution, which is what temperature-only sampling does: the softmax over a language model's logits is a Boltzmann distribution. A new footnote in Sec. 1 says this. We know of no earlier reference for temperature-only sampling and will add any the reviewer suggests.

**"Bonferroni correcting" in Table 1.** Fixed.

**Further experiments.** We added two from sweeps we had already run. Sec. 3 now includes GPQA, the original paper's other benchmark, on 16 models with the same grid as GSM8K; the conclusion is unchanged. An appendix figure reports GSM8K CoT on ten larger models (Qwen 2.5 14B, 32B and 72B, Gemma 2 27B, and Llama 3.1 70B, base and instruct); again min-p does not outperform the other samplers at equal budget. We did not run new open-ended or downstream experiments. The Sec. 6 limitations paragraph now states that GSM8K and GPQA reward accuracy rather than diversity, so they cannot reveal a benefit where diversity is instrumentally useful, such as Minimum Bayes Risk decoding (Freitag et al., 2023), and that open-ended tasks beyond creative writing remain open. The human evaluations and AlpacaEval creative writing evaluations in Secs. 2 and 4 are open-ended tasks, and our re-analysis of them reaches the same conclusion as the benchmark sweeps.

**Title.** Kept for this revision. We revised the body throughout to make the tone more measured (see the General Response).
