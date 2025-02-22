import glob
from typing import Any, Dict
import os
import pprint
import subprocess
import wandb

import src.globals


def run_one_eval():
    run = wandb.init(
        project="min-p-evals",
        config=src.globals.EVAL_DEFAULT_CONFIG,
        entity=wandb.api.default_entity,
    )

    config: Dict[str, Any] = dict(wandb.config)
    pprint.pprint(config)

    do_sample = True if config["temperature"] > 0 else False
    if config["task"] == "gsm8k_cot_llama":
        command = f"""
        lm_eval \
        --model {config['model']} \
        --model_args pretrained={config['model_hf_path']},dtype=auto \
        --batch_size auto \
        --tasks {config['task']} \
        --num_fewshot {config['num_fewshot']} \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path ./lm-eval-output/ \
        --gen_kwargs {config['sampler']}={config['sampler_value']},temperature={config['temperature']},do_sample={do_sample} \
        --device cuda \
        --seed {config['seed']}
        """
        # --wandb_args project=min-p-evals,name={config['task']}_{config['sampler']}_{config['sampler_value']}_temp_{config['temperature']}_{config['model']}_{config['model_args'].replace('/', '_')} \
    else:
        raise NotImplementedError(f"Task {config['task']} is not implemented.")

    command = (
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID"
        f" && export CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
        ' && eval "$(conda shell.bash hook)" && conda activate min_p_env &&' + command
    )

    # Execute the command and block until completion
    print(f"Executing command: {command}")
    process = subprocess.run(
        command, shell=True, check=True, text=True, capture_output=True
    )

    if process.stderr:
        print("Command stderr:")
        print(process.stderr)

    if config["task"] == "gsm8k_cot_llama":
        scores = extract_gsm8k_scores_from_output(process.stdout)
    else:
        raise NotImplementedError(f"Task {config['task']} is not implemented.")

    # Log the results to wandb
    wandb.log(scores)

    wandb.finish()


def extract_gsm8k_scores_from_output(output_text: str) -> Dict[str, float]:
    """Extract exact_match scores from the lm-eval output text."""
    results = {}

    lines = output_text.strip().split("\n")
    for line in lines:
        if "exact_match" in line:
            # Parse the line in the table that contains exact_match
            parts = line.split("|")
            if len(parts) >= 8:
                filter_type = parts[3].strip()
                metric = parts[5].strip()
                value = float(parts[7].strip().split("±")[0])

                # Create a meaningful key
                key = f"{metric}_{filter_type}"
                results[key] = value

    return results


if __name__ == "__main__":
    run_one_eval()
    print("Finished run_one_eval!")
