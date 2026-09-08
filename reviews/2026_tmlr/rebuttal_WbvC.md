# Response to Reviewer WbvC

We thank the reviewer and address both concerns below.

**1. Why min-p warrants re-examination.** We agree that GitHub stars do not establish influence. The revised introduction gives other evidence:

- *Citations.* More than 170 within a year of release.
- *Inference libraries.* Min-p is implemented in Hugging Face Transformers, vLLM, SGLang, llama.cpp, and Text Generation Inference. The integrations followed the paper's results; the ICLR 2025 camera-ready cites them as adoption evidence.
- *Downstream research.* Artificial Hivemind (Jiang et al., NeurIPS 2025 Best Paper) used min-p as its representative decoding-time intervention, found that mode collapse persisted, and concluded that decoding-time interventions in general cannot preserve diversity. That conclusion holds only if min-p is a strong diversity-promoting sampler; our analysis indicates it is not. p-less sampling (Tan et al., ICLR 2026 Oral) used min-p as its primary baseline and evaluated it with the methodology whose flaws we document.
- *Peer review.* 18th highest-scoring submission at ICLR 2025; ICLR Oral.

Re-examination is warranted by a claim's visibility and the work built on it, not by its share of production deployments. The other two reviewers answered "Yes" on audience interest for reasons independent of adoption: samplers affect every LLM use, and platformed papers should hold under scrutiny. We will add any other adoption evidence the reviewer suggests.

**2. Tone.** We revised the manuscript as follows:

- Removed the sentence "For those unfamiliar, AlpacaEval reports win rates..." (Sec. 4.1) and the sentence comparing the revised adoption statement to "publishing a book and then claiming credit for the library" (Sec. 5.2).
- Removed the paragraph "What Went Wrong During the ICLR 2025 Review Process?" (Sec. 6) and the passage in Sec. 5 on why we included the retracted adoption numbers. The factual review record (which claims the meta-review cited) now appears, without commentary, in a short appendix.
- Rewrote editorializing language; each criticism now states what the data or public record show.
- Shortened block quotations; replaced several with inline quotations.
- Added a paragraph to Sec. 6 separating field-wide practices (reporting wins in a dominated region of a trade-off, uncorrected multiple comparisons, indirect LLM-as-a-judge designs) from issues specific to the original paper.

We kept Sec. 5 because the adoption claims were part of the original paper's evidence and its reviewers cited them when recommending acceptance. It now states only the record: the numbers could not be substantiated and were retracted, and the revised statement attributes library usage to min-p.

If the reviewer identifies further passages, we will revise them.
