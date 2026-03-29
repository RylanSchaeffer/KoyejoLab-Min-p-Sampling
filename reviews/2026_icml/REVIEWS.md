# ICML 2026 Reviews

## Turning Down the Heat: A Blueprint for Rigorous Evaluation in Empirical Machine Learning Research

**Submission Number:** 31762
**Authors:** Rylan Schaeffer, Yegor Denisov-Blanch, Joshua Kazdan
**Submitted:** 23 Jan 2026 (modified: 19 Mar 2026)
**Venue:** ICML 2026 Conference Submission

**Primary Area:** Deep Learning -> Large Language Models
**Keywords:** language models, sampling, samplers, min-p, large language models, evaluations, reproducibility, peer review, ML conferences
**LLM Policy:** Policy B
**Reciprocal Reviewing Author:** Yegor Denisov-Blanch

---

## Reviewer KrXT

**Overall Recommendation:** 5 (Accept)
**Confidence:** 4
**Soundness:** 4 (excellent) | **Presentation:** 3 (good) | **Significance:** 4 (excellent) | **Originality:** 4 (excellent)

### Summary

This paper presents a deep investigation into one of the highest scoring submission of ICLR 2025. The claims under questions are regarding the validity of using Min-P sampling for creative and coherent LLM outputs. The authors scrutinize evaluations across benchmarks, LLM-as-a-Judge evaluation, as well as human evaluation conducted as part of the original study. They make a case for better standards for evaluation to improve scientific rigor across the field.

### Strengths And Weaknesses

This is a very unique paper that presents a deep and through investigation. While it is not the first of its kind, in a sea of empirical research work, this paper stands out. Every single claim and every analysis of the paper under investigation has been thoroughly put under microscope. The authors not only raise issues, they also fix them and present the corrected results. Furthermore, direct dialogue which has already led to revisions to the investigated research is a sign of productive and constructive scientific debate.

Further justification and elaboration on the statistical tests selected and utilized could help future readers better understand the reasoning behind the decisions. While the usage of "Best-of-N" sampling for evaluation is interesting, one might suspect fair comparison through hyperparameter search is the main driving factor of such findings.

### Key Questions For Authors

1. Can you further elaborate on selection and justification of the statistical tests selected?
2. How important is the use of Best-of-N evaluation protocol? Earlier work (e.g. Mogrifier LSTM, 2020) have been able to show similar results without the use of it.

### Limitations

Yes

---

## Reviewer 2LLS

**Overall Recommendation:** 3 (Weak reject)
**Confidence:** 4
**Soundness:** 3 (good) | **Presentation:** 2 (fair) | **Significance:** 2 (fair) | **Originality:** 2 (fair)

### Summary

The paper examines the issue of methodological weaknesses in empirical machine learning evaluation that may inflate reported results. As a remedy, it proposes a blueprint for rigorous empirical evaluation, consisting of four standards: (1) fair comparison via controlled hyperparameter volume, (2) valid statistical inference, (3) full methods and data transparency, and (4) consistent reporting.

The main technical contribution is a "Best-of-N" evaluation protocol, which controls for hyperparameter search budget when comparing algorithms. The authors illustrate these standards through a case study re-evaluating a recent high-visibility ICLR 2025 oral paper on the LLM sampling method min-p. They re-evaluate the original experiments by applying their Best-of-N evaluation protocol with extensive hyperparameter sweeps across multiple language models and sampling strategies, showing that the performance advantage reported in the original work disappears when controlling for hyperparameter search volume. The paper further shows that tests on pooled data, misleading visualizations, omitted data, and selective metric reporting contributed to inflated results in the original paper.

Based on this evidence, the authors argue that many state-of-the-art claims in empirical machine learning may be fragile under stricter evaluation standards and advocate adopting their blueprint to ensure that claims are valid.

### Strengths And Weaknesses

Soundness: The paper appears largely technically sound. The re-evaluation of the case-study experiments is rigorous, and the associated claims are supported by experimental results under appropriate evaluation methods. Particularly, the Best-of-N evaluation protocol as main technical contribution is clearly defined and implemented with extensive experiments involving large-scale hyperparameter sweeps across multiple models and sampling methods. The further re-analysis of the case-study experiments applies standard statistical practices and raises plausible concerns about pooled testing, visualization, and reporting choices in the original work. The results provide substantial empirical support for the paper's claims regarding the re-evaluation.

Reproducing the experiments appears feasible based on the provided description, although some implementation details (e.g., seed handling in hyperparameter sweeps, how hyperparameters are sampled) may not be fully specified.

However, the broader methodological claim regarding the fragility of evaluation practices relies primarily on a single case study. As a result, the evidence may be insufficient to fully support this claim. Additional evaluations across multiple papers or experimental settings would strengthen the empirical basis for this argument.

Presentation: The paper is largely well written and easy to follow. The motivation, the proposed standards, and the Best-of-N protocol are explained clearly, and the case-study re-evaluation is presented in a logically structured manner with appropriate figures and experimental descriptions. The overall structure - general motivation, specific case study organized by each standard, general conclusions - provides a clear narrative.

However, large parts of the paper read more like a critique of a specific prior work rather than developing a general framework. Some passages discuss very specific numerical discrepancies, which appear overly detailed for the main narrative, or interactions with the authors of the original paper, which read narrative and somewhat adversarial rather than scientific.

Related work discussion exists but is fragmented and mostly motivational. A comparison to existing evaluation frameworks and benchmarking protocols as well as positioning relative to prior reproducibility papers that already discuss similar issues in ML evaluation is missing. I believe that the paper would benefit from a structured related work section with a systematic survey of evaluation methodology research, particularly on hyperparameter search bias, benchmarking fairness, and statistical evaluation in ML.

Finally, there are several minor presentation issues (typos, overflowing links in the references, oversized figure in appendix) and inconsistencies in the reference format.

Significance: The paper addresses an important and timely issue in empirical machine learning research, namely the reliability of empirical evaluation and the potential for methodological choices like hyperparameter tuning effort, statistical testing, or reporting practices to inflate reported improvements. Both the detailed re-evaluation of a recent high-visibility paper and prior work listed in the introduction illustrate the high relevance of the problem addressed.

However, while the general scope of the work is relatively broad, the demonstrated impact is somewhat limited by the scope of the empirical analysis, which focuses on a single case study involving one specific problem (LLM sampling). While this case study is thorough, the paper draws broader conclusions about evaluation practices in machine learning largely based on this example. A broader analysis across multiple papers, benchmarks, or methodological settings would increase the potential impact of the work.

Furthermore, the practical usefulness of the proposed blueprint may currently be limited as the standards 2-4 are described mainly at a conceptual level rather than being operationalized through concrete procedures or evaluation protocols.

Originality: The paper's main contributions are the proposed blueprint with its 4 standards, the Best-of-N evaluation protocol as the primary technical component, and as a more narrow contribution the re-evaluation of the case study.

While the paper provides a concrete protocol for the first standard of the blueprint, the remaining standards 2-4 are formulated primarily as high-level guidelines that largely reflect established best practices in empirical machine learning research rather than introducing new methodological tools. In particular, the paper does not propose concrete procedures, frameworks, or artifacts like workflows, checklists, or reporting templates that would operationalize these standards.

The Best-of-N evaluation protocol represents the clearest technical contribution. While the core idea is reasonable and clearly demonstrated in the case study, the protocol itself is conceptually simple and closely related to standard hyperparameter sweep analyses that examine performance as a function of tuning effort.

In addition, the protocol lacks proper formalization and further analyses. Particularly, the required compute seems an important limitation of this approach. A comparison to existing protocols with similar purpose is likewise lacking. As a result, the overall novelty of the methodological contribution appears somewhat limited, with the main actionable idea being the Best-of-N protocol and the rest of the blueprint remaining largely conceptual.

As an additional contribution, the paper provides a detailed re-evaluation of the ICLR paper on min-p sampling, identifying several issues in the experimental setup, statistical analysis, and reporting that have inflated the originally reported results. While this analysis is thorough and informative, its novelty is rather moderate and mainly lies in the scale of the experiments and the application of the proposed evaluation protocol rather than in a fundamentally new methodological insight.

Reason: The paper addresses an important issue in empirical ML evaluation and provides a careful re-evaluation of a recent high-visibility study. However, the methodological novelty is limited, the proposed blueprint remains largely conceptual beyond the Best-of-N protocol, and the broader claims rely on a single case study. A proper discussion of related work is missing and the writing style is often more of a critique of the original paper from the case study. Given the limited technical novelty and that the paper is mostly empirical critique and methodological commentary, I have some concerns about the fit with the ICML main track. The work might be more naturally suited to venues or tracks that emphasize reproducibility, benchmarking, or even methodological position papers.

### Key Questions For Authors

1. Is the modified implementation of the case study experiments publicly available?
2. The Best-of-N protocol assumes that algorithms should be compared under equal hyperparameter search effort. While this is a reasonable perspective, other evaluation philosophies exist (e.g., comparing methods at their best achievable performance, considering tuning as part of the method, etc.). Could the authors elaborate why this notion of fairness is preferred and in which settings this is most appropriate?

### Limitations

Yes, the authors explicitly acknowledge the paper's main limitation. Further limitations regarding the proposed Best-of-N protocol, however, are not discussed (see above).

---

## Reviewer dH2p

**Overall Recommendation:** 3 (Weak reject)
**Confidence:** 3
**Soundness:** 3 (good) | **Presentation:** 2 (fair) | **Significance:** 4 (excellent) | **Originality:** 1 (poor)

### Summary

This paper argues that empirical ML research often lacks sufficient rigor and proposes a blueprint consisting of four standards: (1) fair comparison by controlling hyperparameter optimization volume, (2) valid statistical inference, (3) full data transparency, and (4) consistent reporting. The paper applies this blueprint to a case study: the ICLR 2025 oral paper on min-p sampling for LLM decoding. Through extensive sweeps and re-analyses, the authors conclude that min-p does not consistently outperform alternatives once tuning budgets and statistical testing are handled properly.

### Strengths And Weaknesses

**Strengths**

- Addresses an important and timely issue. The paper engages with what could be described as an "evaluation crisis" in empirical ML, highlighting how experimental design and reporting practices can lead to misleading conclusions.
- Solid analysis. The case study appears thorough, involving large sweeps across multiple models, samplers, and temperatures. The paper identifies concrete methodological pitfalls such as unequal tuning budgets, incomplete reporting, and invalid statistical inference.
- Concrete case study. This paper provides a detailed case study examining a widely discussed decoding method, which helps illustrate how the proposed evaluation principles can be applied in practice.

**Weaknesses**

- Limited empirical breadth. The submission centers heavily on a single target paper. While the analysis is detailed, it remains unclear how broadly the proposed blueprint generalizes across other ML evaluation settings. Including additional case studies or smaller demonstrations could strengthen the generality of the proposal.
- Tone and framing. At times the paper reads somewhat adversarial, as if primarily critiquing a specific work, rather than presenting a broadly applicable methodology for the community. The contribution might be strengthened by framing the blueprint more as a general best-practice protocol and illustrating it with multiple examples.
- Unclear methodological contribution beyond best-practice recommendations. Many elements of the proposed blueprint (e.g., careful hyperparameter tuning protocols, statistical testing, and transparent reporting) resemble widely discussed best practices in empirical ML. It would strengthen the paper to clarify what aspects of the framework are genuinely new --- e.g., a formal protocol, measurable criteria, or tools that make these practices easier to implement in practice.

### Key Questions For Authors

1. How should one determine fair hyperparameter search ranges for each method when the methods have qualitatively different parameters or tuning sensitivities? Ensuring comparable optimization budgets across heterogeneous methods seems challenging in practice.
2. The paper reads closer to a position/perspective paper advocating improved evaluation standards. How do the authors see this fitting the expectations of the main research track?

### Limitations

Yes

---

## Reviewer cBMY

**Overall Recommendation:** 2 (Reject)
**Confidence:** 3
**Soundness:** 3 (good) | **Presentation:** 2 (fair) | **Significance:** 3 (good) | **Originality:** 1 (poor)

**Ethics Review Flag:** Yes
**Ethics Expertise Needed:** Responsible Research Practice (e.g., IRB, documentation, research ethics), Privacy and Security (e.g., personally identifiable information), Research Integrity Issues (e.g., plagiarism)

### Summary

The paper "Turning Down the Heat: A Blueprint for Rigorous Evaluation in Empirical Machine Learning Research" discusses a potential crisis of rigor in empirical machine learning research. The authors define a blueprint based on four standards: (1) fair comparison through controlled hyperparameter optimization, (2) valid statistical inference, (3) full data transparency, and (4) consistent reporting. They also propose a "Best-of-N" evaluation protocol. A substantial portion of the paper is devoted to critically reexamining the results of "Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs" (Nguyen et al., 2024), a paper presented at ICLR 2025. The authors reproduced some of the experiments from Nguyen et al. (2024) and also examined the public data associated with that paper, identifying several concerns in the prior publication. These include that manual annotation of human evaluators' qualitative responses does not appear to support the claim that min-p was the preferred sampler, that min-p does not seem to outperform the baselines in quality, diversity, or in a Pareto-optimal trade-off between quality and diversity, and that the claims about community adoption, based on the number of GitHub repositories using min-p and the number of stars across those projects reported in Nguyen et al. (2024), may be substantially overstated.

### Strengths And Weaknesses

Soundness: One of the paper's main contributions is the "Best-of-N" evaluation protocol, which resembles a variant of grid search and, as stated by the authors, is extremely computationally costly: "Verifying this claim required a computational effort significantly larger than the original study. We used the authors' code to conduct an extensive sweep on GSM8K CoT (Cobbe et al., 2021) totaling ~6000 Nvidia A100-hours over the following models, samplers and hyperparameters." It is therefore unclear how feasible this approach would be in many scenarios, particularly when computational resources are more limited. One strength of this paper is that it carefully identifies and documents weaknesses in the paper under discussion, namely "Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs" (Nguyen et al., 2024), which I was not familiar with before reviewing this submission. The authors do a good job of highlighting several important flaws in that work. The new experiments and investigative arguments appear sound, and it is also valuable that the authors contacted Nguyen et al. (2024) to seek clarification. However, one limitation is that the paper relies on a single, albeit real, case study as the main basis for its argument. Beyond this critical reassessment, the paper's broader contribution remains limited. The general recommendations are largely well known and, in many cases, may be seen as standard good research practice. When such practices are not followed, the reasons may reflect deeper structural problems and incentive issues in the field, which are not meaningfully addressed here. As a result, parts of the discussion feel somewhat underdeveloped. Nevertheless, I encourage the authors to continue promoting these standards, which I strongly support, as well as their efforts to revisit and scrutinize already published results.

Presentation: The paper is generally well organized, but the writing is at times repetitive, with some claims referenced multiple times, and there are also a few typos. The paper appears to assume that the reader is already familiar with the min-p work, since much of the text is devoted to a rebuttal of the method presented in (Nguyen et al., 2024). However, it does not sufficiently explain the original motivations, functionality, merits, or applications of min-p, which limits accessibility for readers who are not already familiar with that work. In addition, external links are embedded directly in the text, which reduces readability; they would be clearer if presented explicitly. Several references also contain URL overflows, and Figure 9 appears too large for the page.

Significance: The paper's two main contributions, namely the four critical standards (Standard 1: Fair comparison via controlled hyperparameter optimization volume, Standard 2: Valid statistical inference, Standard 3: Full data transparency, Standard 4: Consistent reporting) and the "Best-of-N" evaluation protocol, are largely well known. As a result, it is difficult to assess how much the discussion presented here, without more concrete guidance on how to implement these principles efficiently and practically, is likely to improve the broader situation. That said, the paper's rebuttal of several issues in "Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs" (Nguyen et al., 2024) may be significant, especially the points concerning LLM-As-A-Judge evaluations being under-specified and indirect, incorrect data pooling in statistical inference, and the fact that manual annotation of human evaluators' qualitative responses does not support the claim that min-p was the preferred sampler. Ideally, further studies should examine the experiments and arguments presented in both papers in order to determine more conclusively whether min-p is a valid and effective method. However, the overall significance of this contribution also depends on the significance of min-p itself.

Originality: The paper's two main contributions, namely the four critical standards and the evaluation protocol, do not appear to be genuinely novel, but rather reflect broadly accepted good practices. Most of the paper is dedicated to rebutting several results and claims from (Nguyen et al., 2024). Because that paper appears to have been influential and widely discussed following ICLR 2025, the new results presented here may still be considered original and relevant.

### Key Questions For Authors

1. Isn't the proposed "Best-of-N" evaluation protocol extremely costly? Could this make its application impractical for many researchers?
2. How does the proposed protocol differ from variations of grid search?
3. If the proposed standards and evaluation protocol are intended for general use, why are they validated on only one case study?
4. How can these four standards and broader lessons be enforced for researchers and reviewers within the current system and under existing incentives?

### Limitations

The authors included a Limitations section and an Impact Statement. The paper reads more like a rebuttal of a previous work, namely (Nguyen et al., 2024), so conventional limitations may not apply in the usual way within this discussion. The main limitation identified by the authors is that, in light of new evidence regarding the cited paper, different conclusions may be warranted. However, with respect to the four research standards and the "Best-of-N" evaluation protocol proposed in this work, the authors do not discuss as a limitation the practical challenge of encouraging or enforcing their adoption in a realistic way.

### Ethical Review Concerns

I flagged this paper for ethics review for the two reasons listed below, as I am unsure whether they may violate any ICML guideline or code of conduct. I do not hold strong opinions on these matters and leave them entirely to the ethics reviewer's discretion. I also believe it is appropriate to state that I do not know the authors of this submission or the authors of (Nguyen et al., 2024), and I was not familiar with the paper "Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs" before starting this review.

1. Although the authors attempt to include some independent contributions, such as four research standards and the "Best-of-N" evaluation protocol, this submission can still be read largely as a public rebuttal of a single paper: "Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs" (Nguyen et al., 2024), which, as the authors explain, "was ranked as the 18th highest-scoring submission to ICLR 2025 and was selected for an Oral presentation." The authors of this submission not only present their own experiments attempting to replicate the results of (Nguyen et al., 2024), but they also report that they were unable to observe the same level of performance. In addition, they include links and quotes from discussions they themselves had with the authors of (Nguyen et al., 2024), which, according to the text, led to substantial changes and retractions in (Nguyen et al., 2024). Moreover, some of the claims could lead readers to infer that the authors of this submission are suggesting that (Nguyen et al., 2024) involved forms of scientific misconduct, such as embellishing results by omitting significant portions of data and fabricating numbers regarding adoption of their method by the open-source community.

2. By including several links in the paper to public chats between themselves and the authors of (Nguyen et al., 2024), the authors of this submission may have inadvertently disclosed their identities, such that the submission can no longer be regarded as anonymous. Specifically, in Section 5.1, the following link is provided: https://github.com/menhguin/minp_paper/issues/6#issuecomment-2686274162, which reveals one of the authors as Mr. Rylan Schaeffer.

---

## Score Summary

| Reviewer | Recommendation | Soundness | Presentation | Significance | Originality | Confidence |
|----------|---------------|-----------|--------------|--------------|-------------|------------|
| KrXT     | 5 (Accept)    | 4         | 3            | 4            | 4           | 4          |
| 2LLS     | 3 (Weak Reject) | 3      | 2            | 2            | 2           | 4          |
| dH2p     | 3 (Weak Reject) | 3      | 2            | 4            | 1           | 3          |
| cBMY     | 2 (Reject)    | 3         | 2            | 3            | 1           | 3          |
