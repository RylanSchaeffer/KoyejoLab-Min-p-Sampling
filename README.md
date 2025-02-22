# Koyejo Lab Min-p Sampling

## Installation

1. (Optional) Update conda:

`conda update -n base -c defaults conda -y`

2. Create and activate the conda environment:

`conda create -n min_p_env python=3.11 -y && conda activate min_p_env`

3. Install the required packages:

`pip install vllm lm_eval wandb pandas seaborn nvidia-htop `

4. If generating the Many-Shot In-Context Learning data:

`conda install pytorch-gpu torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y`

`pip install protobuf sentencepiece`