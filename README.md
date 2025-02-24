# Koyejo Lab Min-p Sampling

## Installation

1. (Optional) Update conda:

`conda update -n base -c defaults conda -y`

2. Create and activate the conda environment:

`conda create -n min_p_env python=3.11 -y && conda activate min_p_env`

3. Install the required packages:

`pip install vllm lm_eval wandb pandas seaborn nvidia-htop`

or exactly install the versions we used:

`conda env create -f environment.yml`

4. Sign into `wandb`: 

`wandb login`

## Running

To evaluate a single model (to sanity check that the code runs), run:

`export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && conda activate min_p_env && python -u scripts/run_one_eval.py`

To run the full evaluation, create a W&B sweep:

And then launch an agent per GPU:

`export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=YOUR GPU NUMBER && conda activate min_p_env && wandb agent ...`