# Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)

**Authors:** Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, Alon Albalak, Yejin Choi
**Venue:** NeurIPS 2025 Datasets & Benchmarks (Best Paper)
**arXiv:** 2510.22954v1

---

## What the paper does

Introduces INFINITY-CHAT, a dataset of 26K diverse open-ended user queries (from WildChat). Documents the "Artificial Hivemind" effect: LM outputs are strikingly homogeneous on open-ended tasks, both within a single model (intra-model repetition) and across different models (inter-model homogeneity). 70+ models tested, 50 samples per query, 31,250 human annotations.

## Min-p findings (directly relevant)

They test min-p (p=0.1, temperature=2.0, top-p=1.0) — an aggressive diversity-seeking configuration. Results (Figure 5):

- Min-p reduces extreme repetition slightly (fewer pairs above 0.9 similarity)
- But **81% of response pairs still exceed 0.7 similarity**
- **61.2% exceed 0.8 similarity**
- Mode collapse persists under min-p

Their conclusion on min-p: "Despite its promise, min-p is not widely adopted, as it is better suited for creative tasks and less effective for close-ended ones. Further, addressing LM repetitiveness through decoding alone places the burden on users to choose the right strategies. Thus, more generalizable solutions are needed at the model training level to robustly preserve output diversity without requiring user intervention."

## Key findings

1. **Intra-model repetition is severe.** With standard sampling (top-p=0.9, T=1.0), 79% of queries produce response pools with avg pairwise similarity > 0.8.
2. **Min-p provides only marginal improvement.** Even with aggressive settings, mode collapse persists.
3. **Inter-model homogeneity is just as bad.** Avg pairwise similarity between responses from *different* models: 71-82%. Verbatim phrase overlaps occur across model families.
4. **The problem is training-level, not decoding-level.** Paper argues decoding-time interventions (including min-p) are fundamentally insufficient.
5. **LM judges and reward models are poorly calibrated for open-ended tasks.** Correlations with human ratings drop to near-zero on similar-quality subsets. This casts doubt on automated quality evaluations (e.g., AlpacaEval) that min-p proponents cite.

## Relevance to our paper

- **Directly undermines min-p's diversity claim** with more rigorous evaluation (real-world open-ended queries, sentence embeddings, 70+ models, 50 samples each) than Nguyen et al.'s narrower benchmarks.
- **Cites Nguyen et al. as reference [60]** and explicitly frames min-p as insufficient.
- **NeurIPS 2025 Best Paper** — high credibility as independent evidence.
- Does NOT use the Best-of-N protocol — their contribution is the dataset and the phenomenon documentation, not a comparative evaluation protocol.
