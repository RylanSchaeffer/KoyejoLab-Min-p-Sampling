# TMLR 2026 Reviews

## Min-p, Max Exaggeration: A Critical Analysis of Min-p Sampling in Language Models

**Submission Number:** 11062
**Submitted:** 03 Aug 2026 (modified: 14 Aug 2026)
**Venue:** Transactions on Machine Learning Research (Under review)
**Submission Type:** Regular submission (no more than 12 pages of main content)
**Assigned Action Editor:** Matt Kusner
**License:** CC BY 4.0

*Note: inline math (e.g., temperature values, significance thresholds) was stripped when copying from OpenReview. Gaps such as "basic ()" or "at ," reflect the missing symbols, not the reviewers' text.*

---

## Reviewer wjeg

**Date:** 05 Sept 2026, 02:02
**Claims supported by accurate, convincing and clear evidence?:** Yes
**Would at least some of TMLR's audience be interested?:** Yes

### Summary Of Contributions

This work comprehensively examines the min-p paper by Nguyen et al. that was accepted to ICLR 2025 as an oral. The examination includes scrutinizing the statistical validity of the original min-p paper findings and points out several flaws in the experimental design. The authors examine the original paper's four lines of evidence:

1. Human evaluations: The human evaluation procedure is reanalyzed. The authors show that scores for the basic sampler were collected but omitted from the methodology, analysis and results of the original paper without mention, and were added to the Camera Ready only after the authors raised it (Sec. 2.1). A more rigorous complementary analysis is added where they run one-sided paired t-tests with a Bonferroni correction and an Intersection-Union Test (IUT). The IUT is helpful because it addresses whether min-p has consistently higher performance than the other sampling methods on all settings, which is the claim the original paper made. After Bonferroni correction for twelve comparisons, only one (quality, min-p versus basic, ) remains significant at , and the Intersection-Union Test fails to reject the null hypothesis. The data provide evidence for a min-p advantage in a single high-temperature setting, but do not support the original claim that min-p consistently outperforms the baselines across all settings.

2. NLP benchmark evaluations: The authors run GSM8K CoT sweeps across nine models, four samplers and 31 temperatures, and show through two complementary best-of-N analyses that min-p does not consistently outperform other samplers once the number of hyperparameters swept is equalized (Sec. 3.1, Appendix B).

3. LLM-as-a-judge evaluations: The authors critique the use of LLM-as-a-judge on three grounds: the methodology is opaque and under-specified (no model, judge, or uncertainty reported), the design compares every sampler against a fixed basic () reference rather than head-to-head, and Table 3(b) reported the higher of two scores for min-p but the lower of two for top-p (Sec. 4.1 to 4.3). They also make a well-placed point that LLM-as-a-judge preferences are likely not transitive, which undermines inference from the indirect design.

4. Community adoption: The authors make two separate points. First, the repository and star counts in the original paper could not be substantiated and were retracted from the Camera Ready (Sec. 5.1). Second, the revised statement attributes the usage of frameworks such as Transformers and vLLM to min-p itself (Sec. 5.2). The wide adoption of these libraries before the min-p merge makes it difficult to separate which portion of any usage number is due to min-p.

**Strengths:** Reanalysis is done on the original paper's own data. The statistical tests are one-sided in min-p's favour and still fail. The benchmark sweep commits substantial compute (~6000 A100-hours) to a fair comparison. The qualitative annotations were released in the original's format. Criticisms draw on the original paper's internal inconsistencies rather than on opinion.

**Weaknesses:** The benchmark extension covers only GSM8K CoT while the min-p paper claim was "across benchmarks." Several claims rest on public exchanges with the original authors that a reader cannot verify from the paper.

Beyond the specific findings, the paper is a careful public reanalysis of a highly ranked accepted work. This kind of accountability should be encouraged and will hopefully motivate the community to be more careful and more precise, and to revise the standards of peer review, starting with ACs that are not susceptible to rewarding hackable metrics.

### Claims And Evidence

Yes. The claims are supported by the evidence for the reasons given in the summary. The human evaluation reanalysis uses the original paper's published data and tests set up to favour min-p. The benchmark comparison controls for the number of hyperparameters swept, which is the right control for a sampler comparison. The LLM-as-a-judge critique rests on two independent arguments, the indirect design and non-transitivity, and the first stands on its own. The community adoption critique is based on widely available data and common sense logic with regards to the nature of large open-source repos.

### Audience Interest

Yes, samplers affect every LLM use. Papers the community has platformed, promoted and set apart as exemplary should hold under scrutiny, and a reanalysis of one of them is of interest beyond the LLM researc community. I think this type of analysis applies to most method research in Machine Learning and science more broadly.

### Requested Changes

1. **Critical:** Reconcile the bold conclusion at the end of Sec. 2.4 ("no apparent advantage") and the abstract with the Sec. 6 statement that the data "weakly suggest" a benefit at higher temperatures. Your own Table 1 has one Bonferroni-surviving result at .

2. **Critical:** In Sec. 4.1, separate the indirect-design argument (comparing every sampler to basic () does not test min-p against top-p) from the non-transitivity argument, which depends on Xu et al. (2025) [1]. The paper the authors cite to support the problem was first posted to arXiv in February and was accepted to ICML 2025, all after the conclusion of the ICLR 2025 review cycle. While the validity of the critique stands, the text should acknowledge the chronology and not add negligence to the existing evidence of indirect design.

3. **Strengthen:** In 3.1, the authors say that the hyperparameters were taken from the paper but "lightly edited". Please explain what that entails. Which values changed, and why change them at all?

4. **Strengthen:** Scope the Sec. 3 conclusion to GSM8K CoT. The original paper claimed superiority "across benchmarks" and GPQA is untested here. The Sec. 3.1 compute caveat should be reflected in the headline.

5. **Strengthen:** The critique in Sec. 2.4 of reporting wins in a dominated region of the quality-diversity trade-off is not specific to min-p and should not be framed as though it were.

6. **Strengthen:** "For those unfamiliar with AlpacaEval" (Sec. 4.1) reads as condescending. Please rephrase.

7. **Strengthen:** The block quotations take too much space. Consider formatting the surrounding white space or moving some to an appendix.

8. **Strengthen:** Please fix the quotation marks to appear correctly in the compiled PDF, e.g. "Llama" and the quotes in Sec. 3.1.

9. **Strengthen:** The last sentence of 5.2 with the library analogy is not necessary, it does not further the understanding of an already simple and logical conclusion from the section and reads as condescending. The tye of scrutiny this paper has provided does not need to rely on figures of speech to send its message across.

[1] Xu, Yi, et al. "Investigating non-transitivity in LLM-as-a-judge." arXiv preprint arXiv:2502.14074 (2025).

### Broader Impact Concerns

The paper should strive to distinguish between issues specific to this work and those common across the broader community, operating in good faith throughout. Claiming a spot on the Pareto frontier based on a region that is practically suboptimal has become common in similar papers. It would strengthen the work for the authors to provide a more in-depth analysis of how community-wide oversights have contributed to murky standards, and to clarify what additional issues have been introduced specifically by the min-p paper itself.

### Additional Comments

This paper --- and this style of doing research more generally --- is necessary and we should uphold these standards. My assessment is limited to whether the paper's claims are supported by the evidence it presents; I cannot anticipate the response of the original min-p authors, and the paper's own Key Limitation in Sec. 6 scopes itself the same way.

---

## Reviewer WbvC

**Date:** 04 Sept 2026, 07:40 (modified: 05 Sept 2026, 02:02)
**Claims supported by accurate, convincing and clear evidence?:** Yes
**Would at least some of TMLR's audience be interested?:** No

### Summary Of Contributions

This paper is a critical analysis of the recently introduced min-p sampling technique and shows that it largely does not work (e.g., that it improves neither quality nor diversity, nor the tradeoff between the two). Here the authors compare min-p sampling against several other sampling techniques: basic (temperature-only) and top-p, over four evaluation strategies: human evaluations, downstream benchmarks, LLM-as-a-judge, and community adoption metrics. Overall, this paper is a very straightforward read.

**Strengths:** I have not heard about min-p sampling prior to reading this paper, but I buy this manuscript's claims that this sampling technique is not significantly better compared to the other sampling techniques in usage today. The experiments were detailed and exhaustive, and carefully described. It is clear to me that substantial effort went into the empirical evaluation.

**Weaknesses:** I have two primary concerns that prevent me from recommending acceptance of the manuscript in its current form:

First, I am not convinced that min-p is sufficiently very widely used or influential for there to be a critical evaluation of the technique; setting aside the min-p authors' claims of adoption in Section 5, no frontier lab has adopted min-p sampling, and most AI users don't really tamper with sampling and probably have also not heard about min-p. In light of this, I would like to see a stronger case on why resolving the empirical merits of this particular sampling method is an impactful research question. My assessment on this could change if there's evidence of broader adoption that I have overlooked.

Second, and this is more important, the tone of this report is too strong and unnecessarily adversarial throughout. It is very uncomfortable to read this paper at times. A critical re-evaluation of previously published empirical claims can certainly constitute a useful scientific contribution, and the authors may be right that some claims in the original min-p paper are unsupported or inaccurate. But this does not warrant scrutinizing the original work or its authors in the manner that the manuscript currently does; e.g., challenging the original authors' claims of community adoption. I would very strongly encourage the authors to moderate their rhetoric throughout the paper and focus more on the actual science and the empirical evidence instead of the min-p authors and the publication of the work.

### Claims And Evidence

I largely buy the empirical claims that this paper makes, and do not have many technical comments.

### Audience Interest

My answer to this is based on the lack of evidence showing that min-p is a widely adopted decoding technique by the broader research community. Again, my mind could be changed on this.

### Requested Changes

As I have mentioned in my contributions summary, I think the authors need to 1. establish that min-p sampling is a widely used technique (e.g., beyond just Github stars) that warrants a critical re-examination, and to 2. improve their presentation of the framing of their evidence.

### Broader Impact Concerns

n/a

### Additional Comments

n/a

---

## Reviewer fy93

**Date:** 25 Aug 2026, 04:06 (modified: 05 Sept 2026, 02:02)
**Claims supported by accurate, convincing and clear evidence?:** Yes
**Would at least some of TMLR's audience be interested?:** Yes

### Summary Of Contributions

The paper provides a critical review of the work "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs". There, it is claimed that a new method called min-p sampling, which is a form of truncation sampling for language generation, can improve both coherence and quality of the generated outputs. In this work, the authors closely examine the published results and data and find that these claims do not hold when, for example, human evaluation accounts for multiple comparisons or free-text human feedback is annotated manually by the authors of the submitted manuscript. Altogether, the authors provide evidence that min-p sampling's performance has been overstated.

### Claims And Evidence

- Overall I find the re-eximation of the data quite convincing and to my best judgement it seems correctly executed.
- The min-p paper is thoroughly checked and the authors have mostly used the existing evaluation data that was published by the authors of that paper and found several inconsistencies.
- The paper is very easy to follow.
- On a different note, I am wondering if it would be possible to run the algorithms on a) newer models (I'm sorry for giving this criticism, as it might seem like a blanket criticism but it would be an interesting finding to see if any of these conclusions have changed) and b) other benchmarks than GSM8k or in a different way on such a benchmark. The reason for the latter is that GSM8k directly might not benefit as much from diversity directly but if the method were used, for example, for Minimum Bayes Risk decoding, the diversity might help improve downstream performance (similar to Freitag et al. 2023). A different dataset might be useful to, for example, understand if there is any benefit in creative writing applications.

Reference: Epsilon Sampling Rocks: Investigating Sampling Strategies for Minimum Bayes Risk Decoding for Machine Translation (Freitag et al., EMNLP Findings 2023)

### Audience Interest

I find these findings very valuable for the community, oftentimes it seems unclear which sampling strategy should be preferred and, while the paper concludes that many strategies perform similarly, it is important the community is not misled by false claims.

### Requested Changes

1. I have never seen the citation of Ackley 1985 for standard sampling and I am uncertain if if directly fits, could this be argued for or changed?
2. Tab. 1: "Bonferroni correcting " -> "Bonferroni correction"
3. If possible, the experiments could be augmented by further experiments that show whether min-p has any downstream advantage or advantage on more open-ended tasks.
4. I personally find the title a bit aggressive but if it stayed it would not significantly change my rating of this work.

### Broader Impact Concerns

No concerns.

---

## Score Summary

TMLR does not use numeric scores. The two acceptance criteria are recorded per reviewer.

| Reviewer | Claims supported? | Audience interested? | Main asks |
|----------|-------------------|----------------------|-----------|
| wjeg     | Yes               | Yes                  | Reconcile Sec. 2.4/abstract with Sec. 6 hedging; separate indirect-design vs. non-transitivity arguments and note chronology of Xu et al.; scope Sec. 3 to GSM8K CoT; explain "lightly edited" hyperparameters; tone and formatting fixes |
| WbvC     | Yes               | No                   | Establish that min-p is widely used enough to warrant re-examination; moderate adversarial tone |
| fy93     | Yes               | Yes                  | Check Ackley 1985 citation; typo in Tab. 1; optional experiments on newer models / open-ended tasks; title is a bit aggressive |
