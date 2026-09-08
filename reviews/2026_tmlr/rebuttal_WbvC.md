# Response to Reviewer WbvC

We thank the reviewer for reading the paper closely and for stating that they "largely buy the empirical claims." The reviewer's two concerns are whether min-p warrants a critical re-examination, and the tone of the manuscript. We address both, and we have revised the paper on both counts.

**1. Why min-p warrants re-examination.** We agree that GitHub stars do not establish influence, which is one of the points of Sec. 5. The case for re-examination rests on other evidence, which the revised introduction now states:

- *Citations.* The original paper accrued more than 170 citations within a year of release.
- *Integration into the inference stack.* Min-p is implemented in Hugging Face Transformers, vLLM, SGLang, llama.cpp and Text Generation Inference. Whether or not most users change their sampler, anyone using these libraries can enable min-p, and several front-ends expose it as a setting. These integrations followed the paper's reported results, and the paper's ICLR 2025 Camera Ready in turn cites the integrations as evidence of adoption.
- *Downstream research built on the claims.* Artificial Hivemind (Jiang et al., NeurIPS 2025 Best Paper) used min-p as its representative decoding-time intervention against output homogeneity, found that mode collapse persisted, and concluded that decoding-time interventions in general cannot preserve diversity. That generalization is warranted only if min-p is a strong diversity-promoting sampler, which our analysis indicates it is not. p-less sampling (Tan et al., ICLR 2026 Oral) adopted min-p as the primary baseline for a new sampler and evaluated it with the same methodology whose flaws we document.
- *Peer-review platforming.* The paper was the 18th highest-scoring submission at ICLR 2025 and received an Oral. The venue itself marked these claims as exemplary.

We also want to reframe the criterion slightly. The value of re-examining a published claim scales with how visible the claim is and how much work builds on it, not with the method's share of production deployments. The two other reviewers reached "Yes" on audience interest for reasons independent of adoption: samplers affect every LLM use, and papers the community has platformed should hold under scrutiny. We hope the evidence above, together with that framing, addresses the reviewer's concern. If the reviewer has a specific form of adoption evidence in mind that we have not considered, we would be glad to add it.

**2. Tone.** We take this seriously and have revised the manuscript accordingly. Concretely:

- We removed the sentence "For those unfamiliar, AlpacaEval reports win rates..." (Sec. 4.1) and the sentence comparing the revised adoption statement to "publishing a book and then claiming credit for the library" (Sec. 5.2).
- We cut commentary on the original authors and on the ICLR 2025 review process that did not carry evidential content, including most of the paragraph "What Went Wrong During the ICLR 2025 Review Process?" and the passage in Sec. 5 explaining why we chose to include the retracted adoption numbers.
- We rewrote editorializing language throughout so that each criticism is a statement of what the data or the public record show.
- We shortened block quotations and replaced several with short inline quotations.
- We added a paragraph to Sec. 6 that separates practices common across the field (reporting wins in a dominated region of a trade-off, uncorrected multiple comparisons, indirect LLM-as-a-judge designs) from issues specific to the original paper.

On the specific example the reviewer gives, the community adoption claims: we kept Sec. 5 because the claims were part of the original paper's evidence for min-p, and reviewers of the original paper cited them when recommending acceptance. The section now states the factual record (the numbers could not be substantiated and were retracted; the revised statement attributes library usage to min-p) without commentary on the authors or the review process.

We would welcome any further specific passages the reviewer finds uncomfortable, and will revise them.
