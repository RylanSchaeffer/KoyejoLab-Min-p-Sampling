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

    if config["sampler"] == "basic":
        gen_kwargs = f"temperature={config['temperature']},do_sample={do_sample}"
    else:
        gen_kwargs = f"{config['sampler']}={config['sampler_value']},temperature={config['temperature']},do_sample={do_sample}"

    if (
        config["task"] == "gsm8k_cot"
        or config["task"] == "gsm8k_cot_llama"
        or config["task"] == "gpqa_main_generative_n_shot"
        or config["task"].startswith("hendrycks_math")
        or config["task"].startswith("mmlu_pro")
    ):
        command = f"""lm_eval \
        --model {config['model']} \
        --model_args pretrained={config['model_hf_path']},dtype=auto \
        --batch_size auto \
        --tasks {config['task']} \
        --num_fewshot {config['num_fewshot']} \
        --log_samples \
        --output_path ./lm-eval-output/ \
        --gen_kwargs {gen_kwargs} \
        --device cuda \
        --seed {config['seed']}
        """
        # --wandb_args project=min-p-evals,name={config['task']}_{config['sampler']}_{config['sampler_value']}_temp_{config['temperature']}_{config['model']}_{config['model_args'].replace('/', '_')} \
    else:
        raise NotImplementedError(f"Task {config['task']} is not implemented.")

    # For the larger models, we will use parallelism.
    if config["model_hf_path"] in {
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-72B-Instruct",
        "google/gemma-2-27b",
        "google/gemma-2-27b-it",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
    }:
        command = command.replace("dtype=auto", "dtype=auto,tensor_parallel_size=2")

    # These models do not have a chat template. All other models should.
    # Use the chat template for all other models.
    # If we don't do this, we'll receive the following error:
    # ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!
    if config["model_hf_path"] not in {
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-70B",
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
    }:
        command = command.rstrip() + " --fewshot_as_multiturn --apply_chat_template"

    command = (
        'eval "$(conda shell.bash hook)" && conda activate min_p_env && ' + command
    )

    # Execute the command and block until completion
    print(f"Executing command: {command}")
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,  # This is what raises the CalledProcessError
            text=True,
            capture_output=True,  # Capture stdout and stderr
        )

        if process.stderr:
            print("Command stderr:")
            print(process.stderr)

        if (
            config["task"] == "gsm8k_cot"
            or config["task"] == "gsm8k_cot_llama"
            or config["task"] == "gpqa_main_generative_n_shot"
            or config["task"].startswith("hendrycks_math")
            or config["task"].startswith("mmlu_pro")
        ):
            print(process.stdout)
            scores = extract_exact_match_scores_from_output(process.stdout)
            # Log the results to wandb
            wandb.log(scores)
        else:
            raise NotImplementedError(f"Task {config['task']} is not implemented.")

    except subprocess.CalledProcessError as e:
        # Handle the error
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")

    except Exception as e:
        # Handle any other exceptions
        print(f"An unexpected error occurred: {str(e)}")
        wandb.log({"error": True, "error_message": str(e), "command": command})
    finally:
        # Always finish the wandb run
        wandb.finish()


def extract_exact_match_scores_from_output(output_text: str) -> Dict[str, float]:
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
