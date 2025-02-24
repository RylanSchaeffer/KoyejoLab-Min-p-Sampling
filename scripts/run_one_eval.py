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
    
    if config["sampler"] == "standard":
        gen_kwargs = f"temperature={config['temperature']},do_sample={do_sample}"
    else:
        gen_kwargs = f"{config['sampler']}={config['sampler_value']},temperature={config['temperature']},do_sample={do_sample}"
    
    if (
        config["task"] == "gsm8k_cot_llama"
        or config["task"] == "gpqa_main_generative_n_shot"
    ):
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
        --gen_kwargs {gen_kwargs} \
        --device cuda \
        --seed {config['seed']}
        """
        # --wandb_args project=min-p-evals,name={config['task']}_{config['sampler']}_{config['sampler_value']}_temp_{config['temperature']}_{config['model']}_{config['model_args'].replace('/', '_')} \
    else:
        raise NotImplementedError(f"Task {config['task']} is not implemented.")

    command = 'eval "$(conda shell.bash hook)" && conda activate min_p_env &&' + command

    # Execute the command and block until completion
    print(f"Executing command: {command}")
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,  # This is what raises the CalledProcessError
            text=True,
            capture_output=True,
        )

        if process.stderr:
            print("Command stderr:")
            print(process.stderr)

        if (
            config["task"] == "gsm8k_cot_llama"
            or config["task"] == "gpqa_main_generative_n_shot"
        ):
            scores = extract_gsm8k_scores_from_output(process.stdout)
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
