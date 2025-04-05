import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pingouin
import scipy.stats
import seaborn as sns

import src.analyze
import src.globals
import src.plot


pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

alpaca_eval_scores_df = pd.read_csv(
    os.path.join(data_dir, "alpacaeval_scores.csv"),
    index_col=None,
)
alpaca_eval_scores_df["Sampling Hyperparameter"] = alpaca_eval_scores_df[
    "Sampling Hyperparameter"
].astype(str)


# Create a figure with 2 subplots
plt.close()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

# First subplot - Countplot
sns.countplot(
    data=alpaca_eval_scores_df,
    x="Sampler",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    ax=ax1,
)
ax1.set_ylabel("Number of Swept Hyperparameters")

# Second subplot - Scatterplot with error bars
# Define offsets based on sampling method
offset_dict = {"Basic": -0.05, "Min-p": 0.0, "Top-p": 0.05}
# Create a copy of the dataframe with adjusted x-coordinates
plotting_df = alpaca_eval_scores_df.copy()
plotting_df["Offset_Temperature"] = plotting_df.apply(
    lambda row: row["Temperature"] + offset_dict.get(row["Sampler"], 0), axis=1
)

# Plot error bars with offset x-coordinates
for method in plotting_df["Sampler"].unique():
    for param in plotting_df["Sampling Hyperparameter"].unique():
        subset = plotting_df[
            (plotting_df["Sampler"] == method)
            & (plotting_df["Sampling Hyperparameter"] == param)
        ]
        if len(subset) > 0:
            ax2.errorbar(
                subset["Offset_Temperature"],
                subset["Length-Controlled Winrate"],
                yerr=1.96 * subset["Standard Error"],
                fmt="none",  # No connecting line
                ecolor="gray",
                alpha=0.5,
            )

# Then add scatter plot with the same offset x-coordinates
g = sns.scatterplot(
    data=plotting_df,
    x="Offset_Temperature",
    y="Length-Controlled Winrate",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    style="Sampling Hyperparameter",
    style_order=sorted(plotting_df["Sampling Hyperparameter"].unique()),
    ax=ax2,
    s=150,
)

# Add dashed horizontal line labeled chance at 50.00.
ax2.axhline(y=50.0, color="gray", linestyle="--", label="Chance")

# Move legend outside plot
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

# Make sure x-axis ticks reflect the original temperature values, not the offset ones
original_temps = sorted(alpaca_eval_scores_df["Temperature"].unique())
ax2.set_xticks(original_temps)
ax2.set_xlim(min(original_temps) - 0.2, max(original_temps) + 0.2)
ax2.set_xlabel("Temperature (Horizontal Offsets Added for Visibility)")
ax2.set_ylabel("Length-Controlled Winrate vs. Basic(Temperature=1.0)")

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined plot
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="combined_samplers_performance_plot",
)
plt.show()

# Use countplot instead of histplot
plt.close()
sns.countplot(
    data=alpaca_eval_scores_df,
    x="Sampler",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
)
plt.ylabel("Number of Swept Hyperparameters")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="countplot_samplers",
)
plt.show()

plt.close()
fig, ax = plt.subplots(figsize=(16, 12))
# Define offsets based on sampling method
offset_dict = {"Basic": -0.05, "Min-p": 0.0, "Top-p": 0.05}  # No offset for Top-p
# Create a copy of the dataframe with adjusted x-coordinates
plotting_df = alpaca_eval_scores_df.copy()
plotting_df["Offset_Temperature"] = plotting_df.apply(
    lambda row: row["Temperature"] + offset_dict.get(row["Sampler"], 0), axis=1
)
# Plot error bars with offset x-coordinates
for method in plotting_df["Sampler"].unique():
    for param in plotting_df["Sampling Hyperparameter"].unique():
        subset = plotting_df[
            (plotting_df["Sampler"] == method)
            & (plotting_df["Sampling Hyperparameter"] == param)
        ]
        if len(subset) > 0:
            plt.errorbar(
                subset["Offset_Temperature"],
                subset["Length-Controlled Winrate"],
                yerr=1.96 * subset["Standard Error"],
                fmt="none",  # No connecting line
                ecolor="gray",
                alpha=0.5,
            )
# Then add scatter plot with the same offset x-coordinates
g = sns.scatterplot(
    data=plotting_df,
    x="Offset_Temperature",
    y="Length-Controlled Winrate",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    style="Sampling Hyperparameter",
    style_order=sorted(plotting_df["Sampling Hyperparameter"].unique()),
    ax=ax,
    s=150,
)
# Add dashed horizontal line labeled chance at 50.00.
plt.axhline(y=50.0, color="gray", linestyle="--", label="Chance")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# Make sure x-axis ticks reflect the original temperature values, not the offset ones
original_temps = sorted(alpaca_eval_scores_df["Temperature"].unique())
plt.xticks(original_temps)
plt.xlim(min(original_temps) - 0.2, max(original_temps) + 0.2)
plt.xlabel("Temperature")
g.set(
    xlabel="Temperature (Horizontal Offsets Added for Visibility)",
    ylabel="Length-Controlled Winrate vs. Basic(Temperature=1.0)",
)
# Save the plot
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=lc_winrate_x=temperature_hue=sampling_method_style=sampling_hyperparameter_with_errors",
)
plt.show()

# plt.close()
# g = sns.scatterplot(
#     data=alpaca_eval_scores_df,
#     x="Temperature",
#     y="Length-Controlled Winrate",
#     hue="Sampler",
#     hue_order=["Basic", "Top-p", "Min-p"],
#     palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
#     style="Sampling Hyperparameter",
# )
# g.set(
#     ylabel="Length-Controlled Winrate vs. Basic(Temperature=1.0)",
# )
# # Add dashed horizontal line labeled chance at 50.00.
# plt.axhline(y=50.0, color="gray", linestyle="--", label="Chance")
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=lc_winrate_x=temperature_hue=sampling_method_style=sampling_hyperparameter",
# )
# plt.show()


print("Finished notebooks/20_alpacaeval")
