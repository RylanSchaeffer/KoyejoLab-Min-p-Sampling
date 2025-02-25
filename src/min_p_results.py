import pandas as pd

# Complete the GPQA Main dataset
min_p_mistral7b_gpqa_results_df = pd.DataFrame(
    [
        ["Standard", "GPQA", 0.7, 0.2723],
        ["Standard", "GPQA", 1.0, 0.2277],
        ["Standard", "GPQA", 1.5, 0.2522],
        ["Standard", "GPQA", 2.0, 0.0580],
        ["Standard", "GPQA", 3.0, 0.0089],
        ["Top-K", "GPQA", 0.7, 0.2634],
        ["Top-K", "GPQA", 1.0, 0.2366],
        ["Top-K", "GPQA", 1.5, 0.2277],
        ["Top-K", "GPQA", 2.0, 0.1652],
        ["Top-K", "GPQA", 3.0, 0.0588],
        ["Top-P", "GPQA", 0.7, 0.2902],
        ["Top-P", "GPQA", 1.0, 0.2500],
        ["Top-P", "GPQA", 1.5, 0.2478],
        ["Top-P", "GPQA", 2.0, 0.0647],
        ["Top-P", "GPQA", 3.0, 0.0046],
        ["Min-P", "GPQA", 0.7, 0.2918],
        ["Min-P", "GPQA", 1.0, 0.2589],
        ["Min-P", "GPQA", 1.5, 0.2813],
        ["Min-P", "GPQA", 2.0, 0.2634],
        ["Min-P", "GPQA", 3.0, 0.2455],
    ],
    columns=["Sampler", "Task", "Temperature", "Accuracy"],
)

# Create the GSM8K CoT dataset
min_p_mistral7b_gsm8k_results_df = pd.DataFrame(
    [
        ["Standard", "GSM8K", 0.7, 0.2956],
        ["Standard", "GSM8K", 1.0, 0.1751],
        ["Standard", "GSM8K", 1.5, 0.0000],
        ["Standard", "GSM8K", 2.0, 0.0000],
        ["Standard", "GSM8K", 3.0, 0.0000],
        ["Top-K", "GSM8K", 0.7, 0.3063],
        ["Top-K", "GSM8K", 1.0, 0.1759],
        ["Top-K", "GSM8K", 1.5, 0.0000],
        ["Top-K", "GSM8K", 2.0, 0.0000],
        ["Top-K", "GSM8K", 3.0, 0.0000],
        ["Top-P", "GSM8K", 0.7, 0.3609],
        ["Top-P", "GSM8K", 1.0, 0.2767],
        ["Top-P", "GSM8K", 1.5, 0.0068],
        ["Top-P", "GSM8K", 2.0, 0.0000],
        ["Top-P", "GSM8K", 3.0, 0.0000],
        ["Min-P", "GSM8K", 0.7, 0.3518],
        ["Min-P", "GSM8K", 1.0, 0.3086],
        ["Min-P", "GSM8K", 1.5, 0.1842],
        ["Min-P", "GSM8K", 2.0, 0.0621],
        ["Min-P", "GSM8K", 3.0, 0.0000],
    ],
    columns=["Sampler", "Task", "Temperature", "Accuracy"],
)

# Optional: Combined dataframe with all results
min_p_mistral7b_combined_results_df = pd.concat(
    [min_p_mistral7b_gpqa_results_df, min_p_mistral7b_gsm8k_results_df]
).reset_index(drop=True)
