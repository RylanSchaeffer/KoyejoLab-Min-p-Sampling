import alpaca_eval
import os
import json
from datasets import load_dataset

# --- Configuration ---

# Set your OpenAI API Key if it's not already an environment variable
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Define the model we want to evaluate
MODEL_TO_EVALUATE = "openai/gpt-oss-20b"
MODEL_NICKNAME = "gpt-oss-20b-creative"  # A custom name for the output folder

# Define the output path for the results
OUTPUT_PATH = f"./alpaca_eval_results/{MODEL_NICKNAME}"

# --- 1. Select Creative Writing Prompts ---

# AlpacaEval uses the tatsu-lab/alpaca_eval dataset.
# There isn't a dedicated "creative writing" subset, so we can
# manually select some prompts that fit the criteria.
print("Loading and filtering for creative prompts...")
full_dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

# Define keywords to identify creative writing prompts
creative_keywords = [
    "story",
    "poem",
    "write a",
    "narrative",
    "creative",
    "imagine",
    "screenplay",
    "dialogue",
    "monologue",
]


def is_creative(instruction):
    """Check if an instruction is likely a creative writing prompt."""
    return any(keyword in instruction.lower() for keyword in creative_keywords)


# Filter the dataset
creative_prompts_dataset = full_dataset.filter(lambda x: is_creative(x["instruction"]))

# AlpacaEval expects a list of dictionaries, so we convert the dataset
creative_prompts = [item for item in creative_prompts_dataset]

# For demonstration purposes, let's use a smaller subset (e.g., the first 10)
# In a real evaluation, you might use the full set.
creative_prompts_subset = creative_prompts[:10]
print(f"Selected {len(creative_prompts_subset)} creative prompts for evaluation.")

# --- 2. Run the Evaluation ---

# The evaluate_from_model function automates the entire process.
# It will:
#   1. Generate outputs for your model on the provided prompts.
#   2. Use an annotator (a powerful LLM like GPT-4) to compare your model's
#      outputs to a reference model's outputs (by default, gpt-4-turbo).
#   3. Calculate a win rate and generate a leaderboard.

print(f"\nStarting evaluation for model: {MODEL_TO_EVALUATE}")
print(f"Results will be saved to: {OUTPUT_PATH}")

# The main evaluation call
leaderboard_df = alpaca_eval.evaluate_from_model(
    model_configs={
        # The key is a unique identifier for your model run
        MODEL_NICKNAME: {
            "model_name": MODEL_TO_EVALUATE,
            # You can add other generation parameters here if needed
            # "temperature": 0.7,
            # "top_p": 1.0,
        }
    },
    # Provide the subset of creative prompts
    eval_dataset=creative_prompts_subset,
    # Specify where to save the outputs, annotations, and leaderboard
    output_path=OUTPUT_PATH,
    # Use a cost-effective and high-quality annotator
    annotators_config="alpaca_eval_gpt4_turbo",
    # Set to True to cache results and avoid re-running completed steps
    is_cache_leaderboard=True,
    # Use a subset of 10 for a quick run
    max_instances=10,
)


# --- 3. Display the Results ---

print("\n--- Evaluation Complete ---")
print("Leaderboard:")
print(leaderboard_df)

# You can also inspect the generated files in the OUTPUT_PATH directory
annotations_path = os.path.join(OUTPUT_PATH, "annotations.json")
model_outputs_path = os.path.join(OUTPUT_PATH, MODEL_NICKNAME, "model_outputs.json")

print(f"\nModel outputs saved to: {model_outputs_path}")
print(f"Annotations (comparisons) saved to: {annotations_path}")

# Example of inspecting the model's output for the first prompt
if os.path.exists(model_outputs_path):
    with open(model_outputs_path, "r") as f:
        model_outputs = json.load(f)
        print("\n--- Example Output from Your Model ---")
        print(f"Instruction: {model_outputs[0]['instruction']}")
        print(f"Output: {model_outputs[0]['output']}")
        print("------------------------------------")
