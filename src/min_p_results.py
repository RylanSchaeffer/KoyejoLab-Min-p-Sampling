import pandas as pd

# Complete the GPQA Main dataset
min_p_mistral7b_gpqa_results_df = pd.DataFrame(
    [
        ["Basic", "GPQA", 0.7, 0.2723],
        ["Basic", "GPQA", 1.0, 0.2277],
        ["Basic", "GPQA", 1.5, 0.2522],
        ["Basic", "GPQA", 2.0, 0.0580],
        ["Basic", "GPQA", 3.0, 0.0089],
        ["Top-k", "GPQA", 0.7, 0.2634],
        ["Top-k", "GPQA", 1.0, 0.2366],
        ["Top-k", "GPQA", 1.5, 0.2277],
        ["Top-k", "GPQA", 2.0, 0.1652],
        ["Top-k", "GPQA", 3.0, 0.0588],
        ["Top-p", "GPQA", 0.7, 0.2902],
        ["Top-p", "GPQA", 1.0, 0.2500],
        ["Top-p", "GPQA", 1.5, 0.2478],
        ["Top-p", "GPQA", 2.0, 0.0647],
        ["Top-p", "GPQA", 3.0, 0.0046],
        ["Min-p", "GPQA", 0.7, 0.2918],
        ["Min-p", "GPQA", 1.0, 0.2589],
        ["Min-p", "GPQA", 1.5, 0.2813],
        ["Min-p", "GPQA", 2.0, 0.2634],
        ["Min-p", "GPQA", 3.0, 0.2455],
    ],
    columns=["Sampler", "Task", "Temperature", "Accuracy"],
)

# Create the GSM8K CoT dataset
min_p_mistral7b_gsm8k_results_df = pd.DataFrame(
    [
        ["Basic", "GSM8K", 0.7, 0.2956],
        ["Basic", "GSM8K", 1.0, 0.1751],
        ["Basic", "GSM8K", 1.5, 0.0000],
        ["Basic", "GSM8K", 2.0, 0.0000],
        ["Basic", "GSM8K", 3.0, 0.0000],
        ["Top-k", "GSM8K", 0.7, 0.3063],
        ["Top-k", "GSM8K", 1.0, 0.1759],
        ["Top-k", "GSM8K", 1.5, 0.0000],
        ["Top-k", "GSM8K", 2.0, 0.0000],
        ["Top-k", "GSM8K", 3.0, 0.0000],
        ["Top-p", "GSM8K", 0.7, 0.3609],
        ["Top-p", "GSM8K", 1.0, 0.2767],
        ["Top-p", "GSM8K", 1.5, 0.0068],
        ["Top-p", "GSM8K", 2.0, 0.0000],
        ["Top-p", "GSM8K", 3.0, 0.0000],
        ["Min-p", "GSM8K", 0.7, 0.3518],
        ["Min-p", "GSM8K", 1.0, 0.3086],
        ["Min-p", "GSM8K", 1.5, 0.1842],
        ["Min-p", "GSM8K", 2.0, 0.0621],
        ["Min-p", "GSM8K", 3.0, 0.0000],
    ],
    columns=["Sampler", "Task", "Temperature", "Accuracy"],
)

# Optional: Combined dataframe with all results
min_p_mistral7b_combined_results_df = pd.concat(
    [min_p_mistral7b_gpqa_results_df, min_p_mistral7b_gsm8k_results_df]
).reset_index(drop=True)
