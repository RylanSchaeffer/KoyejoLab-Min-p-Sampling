# Response to Reviewer dH2p

## General Response

Three of four reviewers note the blueprint relies on a single case study. We now have a second.

**Second case study: p-less sampling.** We applied Best-of-N to p-less (Tan et al., ICLR 2026 Oral), which claims to "consistently outperform." We implemented p-less in vLLM v0.7.3 and are sweeping 28 models on GSM8K CoT and GPQA (5,952 runs, 3 seeds). Best-of-N results will follow within 1-2 days.

P-less has a novel theoretical contribution (Renyi entropy) and real efficiency gains. Our critique targets evaluation, not the method. Independent of Best-of-N, it violates all four standards: default-only baselines (tuned min-p matches p-less on GPQA per their Table 8); no significance tests on accuracy despite having them for efficiency; human eval at T=2.0 vs T=1.0 with 3/6 author annotators; and "consistently outperforms" claimed when min-p wins 2/4 datasets on Llama3-70B, p-less loses 3/4 at T=1.0, and ranks last in creative writing at T=1.0. Different paper, authors, venue: same problems.

**Independent corroboration.** Jiang et al. (Artificial Hivemind, NeurIPS 2025 Best Paper) tested min-p for diversity across 70+ models: 61% of pairs exceeded 0.8 similarity. They over-generalized to "decoding-time interventions are insufficient": based on the unverified assumption that min-p represents decoding-time methods.

**Revisions.** Blueprint and Best-of-N presented first; case studies as illustrations. Adding Related Work (Dodge 2019, Bouthillier 2021, Henderson 2018, Dehghani 2021, Dror 2018, Melis 2020), Algorithm 1 (pseudocode), and operationalized checklist.

---

## Response to Reviewer dH2p

The reviewer rates significance 4/4 but originality 1/4. We take this seriously: the contribution is a diagnostic protocol and the empirical infrastructure to overturn incorrect claims at scale.

**Empirical breadth.** The p-less case study (see General Response) traces a contamination chain: Nguyen et al. (ICLR 2025 Oral) claimed min-p improves quality and diversity based on flawed evaluation. Jiang et al. (NeurIPS 2025 Best Paper) took this at face value and over-generalized. Tan et al. (ICLR 2026 Oral) repeated the same flawed methodology. Three papers, three venues, three author groups: same failures recurring because no verification mechanism exists.

**Originality.** If these standards are well-known, why do two ICLR Orals violate all four? The gap between knowing and enforcing is the contribution. (1) Best-of-N as a diagnostic protocol: like pass@k (Chen et al., 2021), the mechanism (subsampling) is known but repurposing it to answer "does the advantage survive equalized budgets?" is new. Grid search outputs one configuration; Best-of-N outputs comparative curves. Revision adds Algorithm 1. (2) Empirical verification at scale: >6,000 A100-hours to overturn two oral papers. (3) Operationalized checklist validated against both case studies.

**Tone.** Revision restructures: blueprint first, case studies as illustrations. Discrepancies to appendix.

**Q1: Fair ranges for heterogeneous methods.** Best-of-N equalizes *configuration budget* N, not the parameter space. Each method draws N configurations from its natural space. P-less has 1 hyperparameter (temperature); min-p has 2 (temperature x min-p value). At N=20, p-less covers its space more densely: an asymmetry that is *conservative* (favors the richer-parameter method). When baselines match min-p at equal N, min-p had the advantage and still didn't outperform. If a method genuinely needs less tuning, this shows in the curve: faster rise at low N.

**Q2: Venue fit.** Evaluation methodology papers are accepted at top venues: Dodge 2019 (EMNLP), Dehghani 2021 (NeurIPS), Henderson 2018 (AAAI): none proposed new models. Our paper adds: formalized protocol with pseudocode, 6,000+ A100-hours, corrected re-analyses overturning an ICLR Oral, second case study, operationalized checklist. If post-publication verification is excluded from main tracks, the field provides no venue or incentive for this work.
