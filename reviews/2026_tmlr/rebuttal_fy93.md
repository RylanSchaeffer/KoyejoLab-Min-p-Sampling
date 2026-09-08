# Response to Reviewer fy93

We thank the reviewer for the positive assessment and the concrete suggestions.

**Ackley et al. (1985) for standard sampling.** We cite Ackley et al. because, to our knowledge, it is the earliest work to sample discrete states from a temperature-scaled Boltzmann distribution, which is exactly what temperature-only sampling does with a language model's logits (the softmax over logits is a Boltzmann distribution). We added a footnote in Sec. 1 stating this. We are not aware of an earlier reference for temperature-only sampling, and we would gladly add any additional or more standard reference the reviewer suggests.

**"Bonferroni correcting" in Table 1.** Fixed.

**Further experiments: newer models, other benchmarks, downstream and open-ended uses.** We agree these would be informative. We have not added new experiments in this revision; instead the limitations paragraph in Sec. 6 now states that our sweeps cover GSM8K CoT and models released through 2024, that GSM8K rewards accuracy rather than diversity and so cannot reveal a benefit min-p might provide where diversity is instrumentally useful, such as Minimum Bayes Risk decoding (Freitag et al., 2023), and that open-ended tasks beyond the creative writing covered by the human and LLM-as-a-judge evaluations remain open. [GPQA: TO BE FILLED AFTER SWEEP CHECK] We note that the human evaluations and AlpacaEval creative writing evaluations in Secs. 2 and 4 are open-ended tasks, and our re-analysis of them reaches the same conclusion as the benchmark sweeps.

**Title.** We have kept the title for this revision. We appreciate the reviewer noting it would not change their rating, and we have revised the body of the paper throughout to make the tone more measured (see the General Response).
