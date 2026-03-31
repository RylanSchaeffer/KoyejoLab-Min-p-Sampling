# Response to Reviewer 2LLS

## General Response

Three of four reviewers note the blueprint relies on a single case study. We now have a second.

**Second case study: p-less sampling.** We applied Best-of-N to p-less (Tan et al., ICLR 2026 Oral), which claims to "consistently outperform." We implemented p-less in vLLM v0.7.3 and are sweeping 28 models on GSM8K CoT and GPQA (5,952 runs, 3 seeds). Best-of-N results will follow within 1-2 days.

P-less has a novel theoretical contribution (Renyi entropy) and real efficiency gains. Our critique targets evaluation, not the method. Independent of Best-of-N, it violates all four standards: default-only baselines (tuned min-p matches p-less on GPQA per their Table 8); no significance tests on accuracy despite having them for efficiency; human eval at T=2.0 vs T=1.0 with 3/6 author annotators; and "consistently outperforms" claimed when min-p wins 2/4 datasets on Llama3-70B, p-less loses 3/4 at T=1.0, and ranks last in creative writing at T=1.0. Different paper, authors, venue: same problems.

**Independent corroboration.** Jiang et al. (Artificial Hivemind, NeurIPS 2025 Best Paper) tested min-p for diversity across 70+ models: 61% of pairs exceeded 0.8 similarity. They over-generalized to "decoding-time interventions are insufficient": based on the unverified assumption that min-p represents decoding-time methods.

**Revisions.** Blueprint and Best-of-N presented first; case studies as illustrations. Adding Related Work (Dodge 2019, Bouthillier 2021, Henderson 2018, Dehghani 2021, Dror 2018, Melis 2020), Algorithm 1 (pseudocode), and operationalized checklist.

---

## Response to Reviewer 2LLS

**Q1: Code availability.** Yes. All code, sweep configs, W&B data, and notebooks are public at [anonymized URL]. P-less vLLM patch included.

**Q2: Why equal-effort fairness?** The Best-of-N curve serves multiple philosophies: "best achievable" (large N), "tuning as part of the method" (small N; faster rise = easier to tune), "equal effort" (fixed N). A single number collapses this into one gameable point. Limitation: effort = configuration count, not compute.

**Best-of-N formalization.** Grid search outputs one best configuration; Best-of-N outputs comparative performance-vs-budget curves. Same mechanics, different purpose: like pass@k (Chen et al., 2021), the contribution is applying a known mechanism as a diagnostic protocol. Revision adds Algorithm 1, variance-vs-N analysis, and limitations.

**Related work.** Dodge 2019 and Bouthillier 2021 characterize unequal tuning without a reusable protocol. Henderson 2018 shows RL pitfalls without formalization. Melis 2020 does fair comparison ad hoc. Best-of-N systematizes these.

**Standards 2-4 operationalization.** Revision adds a checklist: Standard 2 (significance tests with correction: violated in min-p via pooling, in p-less via absence). Standard 3 (release eval code: p-less released none). Standard 4 (win/loss tables: both overclaimed). Each validated against two papers. Scaling enforcement through automated tools is active work.

**Tone.** Blueprint first; case studies as illustrations. Discrepancies to appendix.

**Venue fit.** Revised paper: formalized protocol with pseudocode, 6,000+ A100-hours across 28 models, corrected re-analyses overturning an ICLR 2025 Oral, second case study, operationalized checklist. Precedent: Henderson 2018 (AAAI), Dodge 2019 (EMNLP), Dehghani 2021 (NeurIPS).

**Cost.** 6,000 hours = sweep cost, not protocol cost. Protocol is post-hoc subsampling: zero additional compute. Even N=5-10 reveals whether claims hold.

**Reproducibility.** Seeds 0-2. Fixed grid: 31 temperatures (0.0-3.0), sampler values per sweep configs. All enumerated, not sampled.
