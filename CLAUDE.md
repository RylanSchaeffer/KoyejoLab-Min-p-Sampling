# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for the paper "Min-p, Max Exaggeration: A Critical Analysis of Min-p Sampling in Language Models" (arXiv:2506.13681). Evaluates min-p sampling against other truncation samplers (top-p, top-k, basic/greedy) across NLP benchmarks and human evaluations. W&B sweeps are publicly available at `rylan/min-p-evals`.

## Environment Setup

```bash
conda create -n min_p_env python=3.11 -y && conda activate min_p_env
pip install vllm lm_eval wandb pandas seaborn nvidia-htop statsmodels
# OR exact reproduction:
conda env create -f environment.yml
# For AlpacaEval:
pip install "alpaca_eval[all]"
```

Key pinned versions: `lm-eval==0.4.7`, `vllm==0.7.3`, `torch==2.5.1`. The lm_eval version is intentionally pinned to 0.4.7 despite a known Gemma 2 templating bug, to avoid introducing confounders mid-sweep.

Login required: `wandb login` and `huggingface-cli login`.

## Running Evaluations

All commands require `PYTHONPATH=.` so `import src.*` resolves.

**Single eval (sanity check):**
```bash
export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && python -u scripts/run_one_eval.py
```

**Full sweeps via W&B:**
```bash
wandb sweep sweeps/nlp_benchmarks/<benchmark>/<config>.yaml
export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=<GPU> && wandb agent rylan/min-p-evals/<sweep_id>
```

**Analysis notebooks (run as scripts, not Jupyter):**
```bash
export PYTHONPATH=. && python notebooks/<notebook_dir>/<notebook_name>.py
```

## Architecture

### `scripts/`
- `run_one_eval.py` — Main evaluation entry point. Invoked by W&B sweep agents. Constructs `lm_eval` CLI commands from W&B config (model, sampler, temperature, seed, task). Instruct models get `--apply_chat_template --fewshot_as_multiturn`; models 14B+ get `tensor_parallel_size=2`. Parses exact_match scores from lm_eval output tables and logs to W&B.
- `run_alpaca_eval.py` — AlpacaEval creative writing evaluation (WIP).

### `src/`
- `globals.py` — Central config: `EVAL_DEFAULT_CONFIG` (default sweep params), display name mappings (`MODELS_NICE_NAMES_DICT`, `SAMPLERS_NICE_NAMES_DICT`, `TASK_NICE_NAMES_DICT`), and ordering lists (`MODELS_ORDER_LIST`, `SAMPLERS_ORDER_LIST`) for consistent plot faceting. When adding a new model or task, update both the name dict and the order list here.
- `analyze.py` — Data pipeline: `download_wandb_project_runs_configs()` fetches W&B sweep results by sweep ID list, maps raw config keys to display names using `globals.py` dicts, and caches to disk as CSV/feather/parquet using MD5-hashed filenames. `compute_best_of_n_avg_scores_df()` and `compute_diff_of_best_of_n_avg_scores_df()` run the best-of-N hyperparameter sweep analysis. `compute_samplers_pairwise_scores_differences_df()` computes pairwise sampler comparisons.
- `plot.py` — LaTeX-rendered matplotlib/seaborn setup (Computer Modern font, `usetex=True`). `save_plot_with_multiple_extensions()` saves both PDF and PNG.
- `min_p_results.py` — Hardcoded baseline results from the original min-p paper (Nguyen et al.) for Mistral 7B on GPQA and GSM8K.

### `notebooks/`
Numbered Python scripts (not .ipynb). Each follows the same pattern:
1. Set `refresh = False` (or `True` to re-download from W&B)
2. Call `src.analyze.setup_notebook_dir()` to create `data/` and `results/` subdirs
3. Call `src.analyze.download_wandb_project_runs_configs()` with hardcoded W&B sweep IDs
4. Produce plots saved to its own `results/` subdirectory

Notebooks: `00-04` = NLP benchmarks (GSM8K, GPQA, MMLU Pro, Hendrycks MATH), `10-12` = human evaluations, `20` = AlpacaEval.

### `sweeps/nlp_benchmarks/`
W&B sweep YAML configs organized by benchmark. Each defines a grid search over models, temperatures (0-3.0), sampler values, and seeds. Naming: `<benchmark>_<sampler>_part<N>.yaml`. All sweeps use `method: grid`, `entity: rylan`, `project: min-p-evals`, and invoke `scripts/run_one_eval.py`.

### `manuscript/`
LaTeX source for the ICML 2026 submission. Main file is `00_main.tex`, with sections split into numbered `.tex` files (`01_introduction.tex` through `07_impact_statement.tex`, `99_appendix.tex`).

### `reviews/`
Peer review materials.
