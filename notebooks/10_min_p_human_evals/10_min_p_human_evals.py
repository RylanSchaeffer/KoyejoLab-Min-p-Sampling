import matplotlib.pyplot as plt
import matplotlib.transforms
import os

import numpy as np
import pandas as pd
import seaborn as sns
from typing import Tuple

import src.analyze
import src.globals
import src.plot


data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# Create rows for all data points in the table
rows = [
    # Temperature 1.0, Low Diversity
    [1.0, "Low", "Min-p", 7.06, 1.48, 5.83, 2.03],
    [1.0, "Low", "Top-p", 5.96, 2.24, 2.40, 2.01],
    # Temperature 1.0, High Diversity
    [1.0, "High", "Min-p", 8.02, 1.35, 7.74, 1.63],
    [1.0, "High", "Top-p", 7.67, 1.38, 7.04, 1.88],
    # Temperature 2.0, Low Diversity
    [2.0, "Low", "Min-p", 7.62, 1.53, 6.91, 1.94],
    [2.0, "Low", "Top-p", 5.43, 2.24, 1.83, 1.61],
    # Temperature 2.0, High Diversity
    [2.0, "High", "Min-p", 7.98, 1.42, 7.96, 1.54],
    [2.0, "High", "Top-p", 7.75, 1.37, 7.66, 1.50],
    # Temperature 3.0, Low Diversity
    [3.0, "Low", "Min-p", 7.74, 1.76, 7.60, 1.86],
    [3.0, "Low", "Top-p", 5.75, 2.33, 2.25, 2.44],
    # Temperature 3.0, High Diversity
    [3.0, "High", "Min-p", 7.57, 1.68, 7.66, 1.45],
    [3.0, "High", "Top-p", 7.11, 2.09, 7.49, 1.74],
]

# Create DataFrame with appropriate column names
human_evals_scores_df = pd.DataFrame(
    rows,
    columns=[
        "Temperature",
        "Diversity",
        "Sampler",
        "Quality (Mean)",
        "Quality (SD)",
        "Diversity (Mean)",
        "Diversity (SD)",
    ],
)

# Compute standard errors.
# Allegedly, 54 annotators: https://github.com/menhguin/minp_paper/issues/4
human_evals_scores_df["Quality (95CI)"] = (
    1.96 * human_evals_scores_df["Quality (SD)"] / np.sqrt(54)
)
human_evals_scores_df["Diversity (95CI)"] = human_evals_scores_df[
    "Diversity (SD)"
] / np.sqrt(54)


def compute_welch_t_test(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Welch's t-test for the given row of the DataFrame."""
    assert len(df) == 2
    top_p_row = df[df["Sampler"] == "Top-p"]
    min_p_row = df[df["Sampler"] == "Min-p"]

    # Compute the t-statistic for quality.
    top_p_quality_mean = top_p_row["Quality (Mean)"].values[0]
    min_p_quality_mean = min_p_row["Quality (Mean)"].values[0]
    top_p_quality_se = top_p_row["Quality (95CI)"].values[0]
    min_p_quality_se = min_p_row["Quality (95CI)"].values[0]
    t_statistic_quality = (top_p_quality_mean - min_p_quality_mean) / np.sqrt(
        top_p_quality_se**2 + min_p_quality_se**2
    )

    top_p_diversity_mean = top_p_row["Diversity (Mean)"].values[0]
    min_p_diversity_mean = min_p_row["Diversity (Mean)"].values[0]
    top_p_diversity_se = top_p_row["Diversity (95CI)"].values[0]
    min_p_diversity_se = min_p_row["Diversity (95CI)"].values[0]
    t_statistic_diversity = (top_p_diversity_mean - min_p_diversity_mean) / np.sqrt(
        top_p_diversity_se**2 + min_p_diversity_se**2
    )

    t_statistics_df = pd.DataFrame(
        {
            "Temperature": [df["Temperature"].values[0]],
            "Diversity": [df["Diversity"].values[0]],
            "Quality t-statistic": [t_statistic_quality],
            "Diversity t-statistic": [t_statistic_diversity],
        },
    )

    return t_statistics_df


# Compute Welch's t-test for Quality and Diversity.
t_statistics = human_evals_scores_df.groupby(["Temperature", "Diversity"]).apply(
    compute_welch_t_test
)

# Create a 2x2 figure:
# Rows: High diversity (top), Low diversity (bottom)
# Columns: Quality vs Temperature, Diversity vs Temperature
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
# Set common labels for the y-axis (Temperature)
for ax in axes[:, 0]:
    ax.set_ylabel("Temperature")
# Define vertical offset based on Sampler type
offsets = {"Min-p": 0.05, "Top-p": -0.05}
colors = {"Min-p": "C0", "Top-p": "C1"}
axes[0, 0].set_title("Quality")
axes[0, 1].set_title("Diversity")
# Loop over the two diversity groups
for i, diversity_level in enumerate(["High", "Low"]):
    # Filter DataFrame for current diversity level
    df_div = human_evals_scores_df[
        human_evals_scores_df["Diversity"] == diversity_level
    ]
    # Plot for each sampler type
    for sampler in ["Min-p", "Top-p"]:
        df_sub = df_div[df_div["Sampler"] == sampler]
        # Calculate vertical offsets
        y_vals = df_sub["Temperature"] + offsets[sampler]
        # Quality plot (left column)
        axes[i, 0].errorbar(
            x=df_sub["Quality (Mean)"],
            y=y_vals,
            xerr=df_sub["Quality (95CI)"],
            fmt="o",
            capsize=5,
            label=f"{sampler}",
            color=colors[sampler],
        )
        axes[i, 0].set_xlim(1, 10)
        # Diversity plot (right column)
        axes[i, 1].errorbar(
            x=df_sub["Diversity (Mean)"],
            y=y_vals,
            xerr=df_sub["Diversity (95CI)"],
            fmt="o",
            capsize=5,
            label=f"{sampler}",
            color=colors[sampler],
        )
        axes[i, 1].set_xlim(1, 10)
    axes[i, 1].text(
        1.1,
        0.5,
        f"Diversity: {diversity_level}",
        transform=axes[i, 1].transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        rotation=-90,
        # bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )
# Remove individual legends (don't call legend() on subplots)
# Instead, create a single legend off to the right.
# One way is to grab the handles and labels from the first subplot:
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, title="Sampler", loc="upper right", bbox_to_anchor=(1.05, 0.95)
)
# Set x-axis labels for both columns
axes[1, 0].set_xlabel("Quality Score")
axes[1, 1].set_xlabel("Diversity Score")
plt.tight_layout(rect=[0, 0, 0.87, 1])  # adjust the layout to make room for the legend
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="min_p_human_evals_scores",
    use_tight_layout=False,
)
# plt.show()

# Create the same figure without the second row.
# Create a 2x2 figure:
# Rows: High diversity (top), Low diversity (bottom)
# Columns: Quality vs Temperature, Diversity vs Temperature
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, squeeze=False)
# Set common labels for the y-axis (Temperature)
for ax in axes[:, 0]:
    ax.set_ylabel("Temperature")
# Loop over the two diversity groups
for i, diversity_level in enumerate(["High"]):
    # Filter DataFrame for current diversity level
    df_div = human_evals_scores_df[
        human_evals_scores_df["Diversity"] == diversity_level
    ]
    # Plot for each sampler type
    for sampler in ["Min-p", "Top-p"]:
        df_sub = df_div[df_div["Sampler"] == sampler]
        # Calculate vertical offsets
        y_vals = df_sub["Temperature"] + offsets[sampler]
        # Quality plot (left column)
        axes[i, 0].errorbar(
            x=df_sub["Quality (Mean)"],
            y=y_vals,
            xerr=df_sub["Quality (95CI)"],
            fmt="o",
            capsize=5,
            label=f"{sampler}",
            color=colors[sampler],
        )
        axes[i, 0].set_xlim(1, 10)
        # Diversity plot (right column)
        axes[i, 1].errorbar(
            x=df_sub["Diversity (Mean)"],
            y=y_vals,
            xerr=df_sub["Diversity (95CI)"],
            fmt="o",
            capsize=5,
            label=f"{sampler}",
            color=colors[sampler],
        )
        axes[i, 1].set_xlim(1, 10)
    axes[i, 1].text(
        1.1,
        0.5,
        f"Diversity: {diversity_level}",
        transform=axes[i, 1].transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        rotation=-90,
        # bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )
# Remove individual legends (don't call legend() on subplots)
# Instead, create a single legend off to the right.
# One way is to grab the handles and labels from the first subplot:
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, title="Sampler", loc="upper right", bbox_to_anchor=(1.05, 0.95)
)
# Set x-axis labels for both columns
axes[0, 0].set_xlabel("Quality Score")
axes[0, 1].set_xlabel("Diversity Score")
plt.tight_layout(rect=[0, 0, 0.87, 1])  # adjust the layout to make room for the legend
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="min_p_human_evals_scores_high_diversity",
    use_tight_layout=False,
)
# plt.show()


print("Finished notebooks/10_min_p_human_evals")
