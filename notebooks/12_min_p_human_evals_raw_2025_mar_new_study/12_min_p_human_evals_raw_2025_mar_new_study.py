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

raw_human_evals_scores_df = pd.read_csv(
    os.path.join(data_dir, "min_p_user_study_v3.0.csv"),
    index_col=0,
)

attentive_check_column = [
    col
    for col in raw_human_evals_scores_df.columns
    if col.startswith("If you're a survey worker")  # Full column name is long.
][0]
# In this v3 version, Minh et al. added a new column titled "Valid Entries".
# Check that it matches what we were doing previously.
assert np.all(
    raw_human_evals_scores_df[attentive_check_column].isna()
    != raw_human_evals_scores_df["Valid Entries"]
)
# Convert from "1"/"0" to True/False for consistency with notebook/11.
raw_human_evals_scores_df["Valid Entries"] = (
    raw_human_evals_scores_df["Valid Entries"] == "1"
)
# Rename the column to re-use our code from notebook/11
raw_human_evals_scores_df.rename(
    columns={
        "Favourite output": "Preferred Sampler",
        "Valid Entries": "Annotator Passed Attention Check",
    },
    inplace=True,
)

# Drop specific columns because those are post-survey questions for human annotators.
raw_human_evals_scores_df.drop(
    columns=raw_human_evals_scores_df.columns[-6:-2].tolist(),
    inplace=True,
)
new_col_names = [
    "Sampler=Basic,Diversity=Low,Temp=1.0,Metric=Quality",
    "Sampler=Basic,Diversity=Low,Temp=1.0,Metric=Diversity",
    "Sampler=Top-p,Diversity=Low,Temp=1.0,Metric=Quality",
    "Sampler=Top-p,Diversity=Low,Temp=1.0,Metric=Diversity",
    "Sampler=Min-p,Diversity=Low,Temp=1.0,Metric=Quality",
    "Sampler=Min-p,Diversity=Low Temp=1.0,Metric=Diversity",
    "Sampler=Basic,Diversity=Low,Temp=2.0,Metric=Quality",
    "Sampler=Basic,Diversity=Low,Temp=2.0,Metric=Diversity",
    "Sampler=Top-p,Diversity=Low,Temp=2.0,Metric=Quality",
    "Sampler=Top-p,Diversity=Low,Temp=2.0,Metric=Diversity",
    "Sampler=Min-p,Diversity=Low,Temp=2.0,Metric=Quality",
    "Sampler=Min-p,Diversity=Low Temp=2.0,Metric=Diversity",
    "Sampler=Basic,Diversity=Low,Temp=3.0,Metric=Quality",
    "Sampler=Basic,Diversity=Low,Temp=3.0,Metric=Diversity",
    "Sampler=Top-p,Diversity=Low,Temp=3.0,Metric=Quality",
    "Sampler=Top-p,Diversity=Low,Temp=3.0,Metric=Diversity",
    "Sampler=Min-p,Diversity=Low,Temp=3.0,Metric=Quality",
    "Sampler=Min-p,Diversity=Low Temp=3.0,Metric=Diversity",
    "Sampler=Basic,Diversity=High,Temp=1.0,Metric=Quality",
    "Sampler=Basic,Diversity=High,Temp=1.0,Metric=Diversity",
    "Sampler=Top-p,Diversity=High,Temp=1.0,Metric=Quality",
    "Sampler=Top-p,Diversity=High,Temp=1.0,Metric=Diversity",
    "Sampler=Min-p,Diversity=High,Temp=1.0,Metric=Quality",
    "Sampler=Min-p,Diversity=High Temp=1.0,Metric=Diversity",
    "Sampler=Basic,Diversity=High,Temp=2.0,Metric=Quality",
    "Sampler=Basic,Diversity=High,Temp=2.0,Metric=Diversity",
    "Sampler=Top-p,Diversity=High,Temp=2.0,Metric=Quality",
    "Sampler=Top-p,Diversity=High,Temp=2.0,Metric=Diversity",
    "Sampler=Min-p,Diversity=High,Temp=2.0,Metric=Quality",
    "Sampler=Min-p,Diversity=High Temp=2.0,Metric=Diversity",
    "Sampler=Basic,Diversity=High,Temp=3.0,Metric=Quality",
    "Sampler=Basic,Diversity=High,Temp=3.0,Metric=Diversity",
    "Sampler=Top-p,Diversity=High,Temp=3.0,Metric=Quality",
    "Sampler=Top-p,Diversity=High,Temp=3.0,Metric=Diversity",
    "Sampler=Min-p,Diversity=High,Temp=3.0,Metric=Quality",
    "Sampler=Min-p,Diversity=High Temp=3.0,Metric=Diversity",
    "Annotator Passed Attention Check",
    "Preferred Sampler",
]
raw_human_evals_scores_df.columns = new_col_names

# Drop the last 8 rows because these are NaNs or population values.
raw_human_evals_scores_df.drop(raw_human_evals_scores_df.index[-8:], inplace=True)

# Create annotator IDs and then drop the index.
raw_human_evals_scores_df["Annotator ID"] = 1 + np.arange(
    len(raw_human_evals_scores_df)
)
raw_human_evals_scores_df.reset_index(inplace=True, drop=True)


attentive_human_annotators_preferred_samplers_df = (
    raw_human_evals_scores_df[
        raw_human_evals_scores_df["Annotator Passed Attention Check"]
    ]
    .groupby("Preferred Sampler")
    .size()
    .reset_index()
    .rename(columns={0: "Num. Human Evaluators"})
)
attentive_human_annotators_preferred_samplers_df[
    "Preferred Sampler"
] = attentive_human_annotators_preferred_samplers_df["Preferred Sampler"].map(
    {
        "a": "Basic",
        "ac": "Basic = Min-p",
        "b": "Top-p",
        "c": "Min-p",
        "-": "None Specified",
    }
)

plt.close()
plt.figure(figsize=(10, 6))
g = sns.barplot(
    data=attentive_human_annotators_preferred_samplers_df,
    x="Num. Human Evaluators",
    y="Preferred Sampler",
    order=["Basic", "Basic = Min-p", "Min-p", "None Specified", "Top-p"],
    hue="Preferred Sampler",
    hue_order=["Basic", "Basic = Min-p", "Min-p", "None Specified", "Top-p"],
)
g.set(
    xlim=(0, 25),
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_human_annotators_preferred_samplers",
)
# plt.show()

# Drop "Preferred Sampler" column. We don't need it moving forward.
raw_human_evals_scores_df.drop(columns=["Preferred Sampler"], inplace=True)

# Pivot the dataframe from wide to tall format.
raw_human_evals_scores_tall_df = raw_human_evals_scores_df.melt(
    id_vars=["Annotator ID", "Annotator Passed Attention Check"],
    var_name="Condition",
    value_name="Score",
)

# Extract Sampler, Diversity, Temp, Metric from Condition.
raw_human_evals_scores_tall_df["Sampler"] = raw_human_evals_scores_tall_df[
    "Condition"
].str.extract(r"Sampler=([\w-]+)")
raw_human_evals_scores_tall_df["Diversity"] = raw_human_evals_scores_tall_df[
    "Condition"
].str.extract(r"Diversity=(\w+)")
raw_human_evals_scores_tall_df["Temperature"] = raw_human_evals_scores_tall_df[
    "Condition"
].str.extract(r"Temp=(\d+\.\d+)")
raw_human_evals_scores_tall_df["Metric"] = raw_human_evals_scores_tall_df[
    "Condition"
].str.extract(r"Metric=(\w+)")

print(
    "Number of Unique Annotators Passing the Attention Check: ",
    raw_human_evals_scores_tall_df[
        raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
    ]["Annotator ID"].nunique(),
)

# Filter the data first
filtered_data = raw_human_evals_scores_tall_df[
    raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
    # & (raw_human_evals_scores_tall_df["Diversity"] == "High")
]
# Make sure filtered_data's "Score" column has type float.
filtered_data["Score"] = filtered_data["Score"].astype(float)

# Create the base barplot using catplot
# Barplot shows mean and 95% CI by default
plt.close()
g = sns.catplot(
    data=filtered_data,
    kind="bar",
    y="Temperature",
    x="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    row="Diversity",
    row_order=["High", "Low"],
    col="Metric",
    margin_titles=True,
    order=[3.0, 2.0, 1.0],
)

# Overlay the stripplot on each facet's axes
# Iterate through the axes dictionary provided by catplot
for (row_val, col_val), ax in g.axes_dict.items():
    # Filter data for the specific facet
    facet_data = filtered_data[
        (filtered_data["Metric"] == col_val) & (filtered_data["Diversity"] == row_val)
    ]
    sns.stripplot(
        data=facet_data,
        y="Temperature",
        x="Score",
        hue="Sampler",
        hue_order=["Basic", "Top-p", "Min-p"],
        # Use a different palette (e.g., grayscale) or marker for strips
        # to distinguish them from the bars, or make them subtle.
        palette=["#555555"] * len(src.globals.SAMPLERS_ORDER_LIST),  # Dark gray points
        order=[3.0, 2.0, 1.0],
        ax=ax,
        dodge=True,  # Dodge points along the categorical axis like the bars
        jitter=True,  # Add jitter to prevent points overlapping perfectly
        size=3,  # Make points smaller
        alpha=0.5,  # Make points semi-transparent
    )
    # Remove the legend created by stripplot on each subplot
    # We'll keep the main legend created by catplot's barplot
    if ax.legend_:
        ax.legend_.remove()


g.set(xlim=(0.5, 10.5))
# By default, y axis goes from top to bottom: 1.0, 2.0, 3.0.
# We want 3.0 on top and 1.0 on bottom.
# Uncomment if you need to invert y-axis
# for ax in g.axes.flat:
#     ax.invert_yaxis()

# Move the original legend created by catplot (from the bars)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=temp_x=score_hue=sampler_row=diversity_col=metric_barnstrip",  # Updated filename
)
# plt.show()

# Print averages to confirm numerical values against Nguyen et al. (2024)'s Table 15.
mean_and_sem_scores_for_all_conditions_df = (
    filtered_data.groupby(["Diversity", "Metric", "Temperature", "Sampler"])
    .agg({"Score": ["mean", "sem"]})["Score"]
    .reset_index()
)
print(mean_and_sem_scores_for_all_conditions_df)

pivoted_mean_and_sem_scores_for_all_conditions_df = (
    mean_and_sem_scores_for_all_conditions_df.pivot_table(
        index=["Diversity", "Temperature", "Sampler"],
        columns=["Metric"],
        values=["mean", "sem"],
    )
)
pivoted_mean_and_sem_scores_for_all_conditions_df.columns = [
    f"{metric} {stat}"
    for metric, stat in pivoted_mean_and_sem_scores_for_all_conditions_df.columns
]
pivoted_mean_and_sem_scores_for_all_conditions_df.reset_index(inplace=True, drop=False)
pivoted_mean_and_sem_scores_for_all_conditions_df["ci95 Quality"] = (
    1.96 * pivoted_mean_and_sem_scores_for_all_conditions_df["sem Quality"]
)
pivoted_mean_and_sem_scores_for_all_conditions_df["ci95 Diversity"] = (
    1.96 * pivoted_mean_and_sem_scores_for_all_conditions_df["sem Diversity"]
)

plt.close()
g = sns.relplot(
    data=pivoted_mean_and_sem_scores_for_all_conditions_df,
    kind="scatter",
    x="mean Quality",
    y="mean Diversity",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    col="Diversity",
    col_order=["Low", "High"],
    style="Temperature",
    s=100,
    linewidth=0,
)
hue_order = ["Basic", "Top-p", "Min-p"]
color_map = dict(zip(hue_order, sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST))))
for col_name, ax in zip(g.col_names, g.axes.flat):
    facet_data = pivoted_mean_and_sem_scores_for_all_conditions_df[
        pivoted_mean_and_sem_scores_for_all_conditions_df["Diversity"] == col_name
    ]
    # 4. Iterate through each row in this filtered data subset
    for _, row in facet_data.iterrows():
        # Get the sampler for this specific point to determine the color
        sampler_value = row["Sampler"]
        color = color_map.get(
            sampler_value, "black"
        )  # Default to black if sampler not in map
        # 5. Plot the error bar on the current axis 'ax'
        ax.errorbar(
            x=row["mean Quality"],
            y=row["mean Diversity"],
            xerr=row["ci95 Quality"],
            yerr=row["ci95 Diversity"],
            fmt="none",  # Don't plot a marker or line for the error bar point
            ecolor=color,  # Use the color corresponding to the sampler
            elinewidth=2,  # Adjust line width as needed
            capsize=3,  # Adjust cap size as needed
            alpha=0.7,  # Optional: make error bars slightly transparent
        )
# g.axes.flat[0].axhline(
#     y=7.5, xmin=0.0, xmax=0.725, linestyle="--", color="black", alpha=0.5
# )
# g.axes.flat[0].axvline(
#     x=7.75, ymin=0.0, ymax=0.7, linestyle="--", color="black", alpha=0.5
# )
g.set(
    xlabel="Quality Score",
    ylabel="Diversity Score",
    xlim=(0.5, 10.5),
    ylim=(0.5, 10.5),
)
g.set_titles(col_template="Diversity Setting = {col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=diversity_score_x=quality_score_hue=sampler_col=diversity_style=temp",  # Updated filename
)
# plt.show()
print("Finished notebooks/12_min_p_human_evals_raw_2025_mar_new_study!")
