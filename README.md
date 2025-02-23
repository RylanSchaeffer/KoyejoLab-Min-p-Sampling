# Koyejo Lab Min-p Sampling

## Installation

1. (Optional) Update conda:

`conda update -n base -c defaults conda -y`

2. Create and activate the conda environment:

`conda create -n min_p_env python=3.11 -y && conda activate min_p_env`

3. Install the required packages:

`pip install vllm lm_eval wandb pandas seaborn nvidia-htop`

4. Sign into `wandb`: 

`wandb login`

## Running

To evaluate a single model (to sanity check that the code runs), run:

`export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && python -u scripts/run_one_eval.py`

To run the full evaluation, create a W&B sweep: