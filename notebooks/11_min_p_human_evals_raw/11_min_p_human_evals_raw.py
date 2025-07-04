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

# Columns:
# 00 = {str} 'Prompt: "Write me a creative story?"\r\n\r\n3 output samples are presented per model/settings combo\r\n\r\nModel A outputs: \r\n\r\nSample 1: \r\nOnce upon a time, in a world beyond our own, there existed a mystical realm known as Aethoria – a land of breathtaking beau
# 01 = {str} 'Model A output diversity (how different/distinct the 3 stories feel from each other)'
# 02 = {str} 'Model B output quality'
# 03 = {str} 'Model B output diversity'
# 04 = {str} 'Model C output quality'
# 05 = {str} 'Model C output diversity'
# 06 = {str} 'Prompt: "Write me a creative story?"\r\n\r\n3 output samples are presented per model/settings combo\r\n\r\nModel A outputs: \r\n\r\nSample 1: \r\nDeep in the Whimsy Woods, where flowers glowed with an ethereal light and the air hummed a soft melody, there lived an ench
# 07 = {str} 'Model A output diversity (how different/distinct the 3 stories feel from each other).1'
# 08 = {str} 'Model B output quality.1'
# 09 = {str} 'Model B output diversity.1'
# 10 = {str} 'Model C output quality.1'
# 11 = {str} 'Model C output diversity.1'
# 12 = {str} 'Prompt: "Write me a creative story?"\r\n\r\n3 output samples are presented per model/settings combo\r\n\r\nModel A outputs: \r\n\r\nSample 1: \r\nOnce upon a time, in a tiny village surrounded only by unrelenting stretches of dark mystic forest -and an ancient castle u
# 13 = {str} 'Model A output diversity (how different/distinct the 3 stories feel from each other).2'
# 14 = {str} 'Model B output quality.2'
# 15 = {str} 'Model B output diversity.2'
# 16 = {str} 'Model C output quality.2'
# 17 = {str} 'Model C output diversity.2'
# 18 = {str} 'Prompt: "Write me a creative story?"\r\n\r\n3 output samples are presented per model/settings combo\r\n\r\nModel A outputs: \r\n\r\nSample 1: \r\nIn the quaint village of Luminaria, nestled between two great mountains, a mystical phenomenon occurred every night. As the
# 19 = {str} 'Model A output diversity (how different/distinct the 3 stories feel from each other).3'
# 20 = {str} 'Model B output quality.3'
# 21 = {str} 'Model B output diversity.3'
# 22 = {str} 'Model C output quality.3'
# 23 = {str} 'Model C output diversity.3'
# 24 = {str} 'Prompt: "Write me a creative story?"\r\n\r\n3 output samples are presented per model/settings combo\r\n\r\nModel A outputs: \r\n\r\nSample 1: \r\nDeep within the heart of a vibrant, mystifying forest, where moonlight filtering through canopy formed glowing stars on an
# 25 = {str} 'Model A output diversity (how different/distinct the 3 stories feel from each other).4'
# 26 = {str} 'Model B output quality.4'
# 27 = {str} 'Model B output diversity.4'
# 28 = {str} 'Model C output quality.4'
# 29 = {str} 'Model C output diversity.4'
# 30 = {str} 'Prompt: "Write me a creative story?"\r\n\r\n3 output samples are presented per model/settings combo\r\n\r\nModel A outputs: \r\n\r\nSample 1: \r\nOnce upon a world far larger than yours. Deep beneath a planet of ice, water cascader down massive cavern, shaping stones s
# 31 = {str} 'Model A output diversity (how different/distinct the 3 stories feel from each other).5'
# 32 = {str} 'Model B output quality.5'
# 33 = {str} 'Model B output diversity.5'
# 34 = {str} 'Model C output quality.5'
# 35 = {str} 'Model C output diversity.5'
# 36 = {str} 'Which Model(s) on which Settings did you like the most overall? What did you like about it?\r\nPlease explain in at least 1-2 sentences.\r\nWe may provide a bonus for: particularly helpful descriptions, close reading and detailed feedback about what you did a
# 37 = {str} "Which AI chatbots do you regularly use, if any (e.g. ChatGPT, Claude, Gemini)? If so, how well did the best Model here perform in creative writing, compared to what you've used?"
# 38 = {str} 'Any other comments/anything that stood out to you?'
# 39 = {str} "If you're a survey worker (e.g. from Prolific), please enter your Worker ID in the correct format below. NOTE: THIS HAS BEEN ANONYMISED\r\n\r\n (Thank you for your time and careful attention!  don't forget to click the completion link after you submit!)  "
raw_human_evals_scores_df = pd.read_csv(
    os.path.join(data_dir, "min_p_user_study_v2.0_rylan_annotated.csv"),
    index_col=0,
)

attentive_check_column = [
    col
    for col in raw_human_evals_scores_df.columns
    if col.startswith("If you're a survey worker")  # Full column name is long.
][0]
raw_human_evals_scores_df[
    "Annotator Passed Attention Check"
] = ~raw_human_evals_scores_df[attentive_check_column].isna()

# Drop specific columns because those are post-survey questions for human annotators.
raw_human_evals_scores_df.drop(
    columns=raw_human_evals_scores_df.columns[-6:-2],
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
    "Rylan's Annotations of Which Model(s) Were Most Preferred",
    "Annotator Passed Attention Check",
]
raw_human_evals_scores_df.columns = new_col_names

# Drop the last 13 rows because these are NaNs or population values.
raw_human_evals_scores_df.drop(raw_human_evals_scores_df.index[-13:], inplace=True)

# Create annotator IDs and then drop the index.
raw_human_evals_scores_df["Annotator ID"] = 1 + np.arange(
    len(raw_human_evals_scores_df)
)
raw_human_evals_scores_df.reset_index(inplace=True, drop=True)

rylans_annotations_df = raw_human_evals_scores_df[
    [
        "Annotator Passed Attention Check",
        "Rylan's Annotations of Which Model(s) Were Most Preferred",
    ]
].copy()


def compute_preferred_sampler(rylans_annotation: str) -> str:
    if not isinstance(rylans_annotation, str):
        return "None Specified"

    samplers_list = []
    if "Sampler=Basic" in rylans_annotation:
        samplers_list.append("Basic")
    if "Sampler=Top-p" in rylans_annotation:
        samplers_list.append("Top-p")
    if "Sampler=Min-p" in rylans_annotation:
        samplers_list.append("Min-p")

    if len(samplers_list) == 0:
        return "None Specified"
    else:
        return " = ".join(samplers_list)


rylans_annotations_df["Preferred Sampler"] = rylans_annotations_df[
    "Rylan's Annotations of Which Model(s) Were Most Preferred"
].map(compute_preferred_sampler)

attentive_human_annotators_preferred_samplers_df = (
    rylans_annotations_df[rylans_annotations_df["Annotator Passed Attention Check"]]
    .groupby("Preferred Sampler")
    .size()
    .reset_index()
    .rename(columns={0: "Num. Human Evaluators"})
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
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_human_annotators_preferred_samplers",
)
# plt.show()

# Remove Rylan's annotations for subsequent analysis.
raw_human_evals_scores_df.drop(
    columns=["Rylan's Annotations of Which Model(s) Were Most Preferred"],
    inplace=True,
)

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

# Compute basic statistics.
basic_statistics_df = (
    raw_human_evals_scores_tall_df[
        raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
    ]
    .groupby(["Sampler", "Diversity", "Temperature", "Metric"])["Score"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
print(basic_statistics_df)


raw_human_evals_subset_scores_tall_df = raw_human_evals_scores_tall_df[
    raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
    & (raw_human_evals_scores_tall_df["Diversity"] == "High")
]

# Note: The KS test assumes independence, which is violated in this data
# because the same annotator rates multiple samplers' outputs.
two_sample_ks_test_results_dfs_list = []
paired_t_test_results_dfs_list = []
wilcoxon_test_results_dfs_list = []
for (metric, temperature), grouped_df in raw_human_evals_subset_scores_tall_df.groupby(
    ["Metric", "Temperature"]
):
    # Shape: (num_annotators, num_samplers)
    pivoted_grouped_df = grouped_df.pivot(
        index="Annotator ID",
        columns="Sampler",
        values="Score",
    )

    # Test if Min-p is greater than Basic under a two-sample Kolmogorov-Smirnov test.
    two_sample_ks_test_result = scipy.stats.ks_2samp(
        pivoted_grouped_df["Min-p"],
        pivoted_grouped_df["Basic"],
        alternative="greater",
    )
    two_sample_ks_test_result_df = pd.DataFrame(
        {
            "Metric": metric,
            "Temperature": temperature,
            "Comparison": "Min-p > Basic",
            "KS Statistic": two_sample_ks_test_result.statistic,
            "p-val": two_sample_ks_test_result.pvalue,
        },
        index=[0],
    )
    two_sample_ks_test_results_dfs_list.append(two_sample_ks_test_result_df)

    # Test if Min-p is greater than Basic under a paired t-test.
    min_p_gt_standard_paired_t_test_results_df = pingouin.ttest(
        x=pivoted_grouped_df["Min-p"],
        y=pivoted_grouped_df["Basic"],
        paired=True,
        alternative="greater",
    )
    min_p_gt_standard_paired_t_test_results_df["Metric"] = metric
    min_p_gt_standard_paired_t_test_results_df["Temperature"] = temperature
    min_p_gt_standard_paired_t_test_results_df["Comparison"] = "Min-p > Basic"
    paired_t_test_results_dfs_list.append(min_p_gt_standard_paired_t_test_results_df)

    # Test if Min-p is greater than Basic under a Wilcoxon signed-rank test.
    min_p_gt_standard_wilcoxon_test_results_df = pingouin.wilcoxon(
        x=pivoted_grouped_df["Min-p"],
        y=pivoted_grouped_df["Basic"],
        alternative="greater",
    )
    min_p_gt_standard_wilcoxon_test_results_df["Metric"] = metric
    min_p_gt_standard_wilcoxon_test_results_df["Temperature"] = temperature
    min_p_gt_standard_wilcoxon_test_results_df["Comparison"] = "Min-p > Basic"
    wilcoxon_test_results_dfs_list.append(min_p_gt_standard_wilcoxon_test_results_df)

    # Test if Min-p is greater than Top-p under a two-sample Kolmogorov-Smirnov test.
    two_sample_ks_test_result = scipy.stats.ks_2samp(
        pivoted_grouped_df["Min-p"],
        pivoted_grouped_df["Top-p"],
        alternative="greater",
    )
    two_sample_ks_test_result_df = pd.DataFrame(
        {
            "Metric": metric,
            "Temperature": temperature,
            "Comparison": "Min-p > Top-p",
            "KS Statistic": two_sample_ks_test_result.statistic,
            "p-val": two_sample_ks_test_result.pvalue,
        },
        index=[0],
    )
    two_sample_ks_test_results_dfs_list.append(two_sample_ks_test_result_df)

    # Test if Min-p is greater than Top-p under a paired t-test.
    min_p_gt_top_p_ttest_results_df = pingouin.ttest(
        x=pivoted_grouped_df["Min-p"],
        y=pivoted_grouped_df["Top-p"],
        paired=True,
        alternative="greater",
    )
    min_p_gt_top_p_ttest_results_df["Metric"] = metric
    min_p_gt_top_p_ttest_results_df["Temperature"] = temperature
    min_p_gt_top_p_ttest_results_df["Comparison"] = "Min-p > Top-p"
    paired_t_test_results_dfs_list.append(min_p_gt_top_p_ttest_results_df)

    # Test if Min-p is greater than Top-p under a Wilcoxon signed-rank test.
    min_p_gt_top_p_wilcoxon_test_results_df = pingouin.wilcoxon(
        x=pivoted_grouped_df["Min-p"],
        y=pivoted_grouped_df["Top-p"],
        alternative="greater",
    )
    min_p_gt_top_p_wilcoxon_test_results_df["Metric"] = metric
    min_p_gt_top_p_wilcoxon_test_results_df["Temperature"] = temperature
    min_p_gt_top_p_wilcoxon_test_results_df["Comparison"] = "Min-p > Top-p"
    wilcoxon_test_results_dfs_list.append(min_p_gt_top_p_wilcoxon_test_results_df)

# Concatenate the results.
two_sample_ks_test_results_df = pd.concat(
    two_sample_ks_test_results_dfs_list
).reset_index(
    drop=True,
)
paired_t_test_results_df = pd.concat(paired_t_test_results_dfs_list).reset_index(
    drop=True,
)
wilcoxon_test_results_df = pd.concat(wilcoxon_test_results_dfs_list).reset_index(
    drop=True,
)

# Correct for multiple comparisons.
two_sample_ks_test_results_df["p-val (Bonferroni Corrected)"] = pingouin.multicomp(
    pvals=two_sample_ks_test_results_df["p-val"],
    alpha=0.05,
    method="bonf",
)[1]
paired_t_test_results_df["p-val (Bonferroni Corrected)"] = pingouin.multicomp(
    pvals=paired_t_test_results_df["p-val"],
    alpha=0.05,
    method="bonf",
)[1]
wilcoxon_test_results_df["p-val (Bonferroni Corrected)"] = pingouin.multicomp(
    pvals=wilcoxon_test_results_df["p-val"],
    alpha=0.05,
    method="bonf",
)[1]

# Reorder the columns to make reading them easier.
paired_t_test_results_df = paired_t_test_results_df[
    [
        "Metric",
        "Temperature",
        "Comparison",
        "alternative",
        "p-val",
        "T",
        "power",
        "dof",
        "CI95%",
        "BF10",
        "p-val (Bonferroni Corrected)",
    ]
]
wilcoxon_test_results_df = wilcoxon_test_results_df[
    [
        "Metric",
        "Temperature",
        "Comparison",
        "alternative",
        "p-val",
        "p-val (Bonferroni Corrected)",
        "W-val",
        "RBC",
        "CLES",
    ]
]
print("Paired T-Test Results: \n", paired_t_test_results_df)
print("Paired Wilcoxon Signed-Rank Test Results: \n", wilcoxon_test_results_df)

# Save results to disk.
two_sample_ks_test_results_df.to_csv(
    os.path.join(results_dir, "two_sample_ks_test_results.csv")
)
paired_t_test_results_df.to_csv(os.path.join(results_dir, "paired_t_test_results.csv"))
wilcoxon_test_results_df.to_csv(os.path.join(results_dir, "wilcoxon_test_results.csv"))

plt.close()
g = sns.displot(
    data=raw_human_evals_scores_tall_df[
        raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
        & (raw_human_evals_scores_tall_df["Diversity"] == "High")
    ],
    kind="kde",
    x="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    col="Metric",
    row="Temperature",
    row_order=["3.0", "2.0", "1.0"],
    common_norm=False,  # Normalize each hue separately
    clip=(1, 10),  # Clip the density to [1, 10] because exact match.
    facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
)
g.set(xlim=(1, 10))
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=density_x=score_hue=sampler_row=temp_col=metric_kde",
)
# plt.show()

plt.close()
g = sns.displot(
    data=raw_human_evals_scores_tall_df[
        raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
        & (raw_human_evals_scores_tall_df["Diversity"] == "High")
    ],
    kind="ecdf",
    x="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    col="Metric",
    row="Temperature",
    row_order=["3.0", "2.0", "1.0"],
    facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
)
g.set(xlim=(1, 10), ylabel="Empirical CDF")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=density_x=score_hue=sampler_row=temp_col=metric_ecdf",
)
# plt.show()

plt.close()
g = sns.catplot(
    data=raw_human_evals_scores_tall_df[
        raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
        & (raw_human_evals_scores_tall_df["Diversity"] == "High")
    ],
    kind="point",
    y="Temperature",
    x="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    col="Metric",
    row="Diversity",
    margin_titles=True,
    order=[3.0, 2.0, 1.0],
)
g.set(xlim=(1, 10))
# By default, y axis goes from top to bottom: 1.0, 2.0, 3.0.
# We want 3.0 on top and 1.0 on bottom.
# for ax in g.axes.flat:
#     ax.invert_yaxis()
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=temp_x=score_hue=sampler_row=diversity_col=metric_point",
)
# plt.show()

plt.close()
g = sns.catplot(
    data=raw_human_evals_scores_tall_df[
        raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
        & (raw_human_evals_scores_tall_df["Diversity"] == "High")
    ],
    kind="box",
    y="Temperature",
    x="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    col="Metric",
    row="Diversity",
    margin_titles=True,
    order=[3.0, 2.0, 1.0],
)
g.set(xlim=(0.5, 10.5))
# By default, y axis goes from top to bottom: 1.0, 2.0, 3.0.
# We want 3.0 on top and 1.0 on bottom.
# for ax in g.axes.flat:
#     ax.invert_yaxis()
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=temp_x=score_hue=sampler_row=diversity_col=metric_boxnwhisker",
)
# plt.show()

# Filter the data first
filtered_data = raw_human_evals_scores_tall_df[
    raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
    & (raw_human_evals_scores_tall_df["Diversity"] == "High")
]

# Create the base stripplot using catplot
plt.close()
g = sns.catplot(
    data=filtered_data,
    kind="strip",  # Changed from "box" to "strip"
    y="Temperature",
    x="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    row="Metric",
    col="Diversity",
    margin_titles=True,
    order=[3.0, 2.0, 1.0],
    # Optional: Adjust strip plot parameters for better visibility if needed
    # jitter=True,
    # dodge=True,
    # alpha=0.7
)

# Overlay the pointplot on each facet's axes
# We iterate through the axes dictionary provided by catplot
for (row_val, col_val), ax in g.axes_dict.items():
    # Filter data for the specific facet
    facet_data = filtered_data[
        (filtered_data["Metric"] == row_val) & (filtered_data["Diversity"] == col_val)
    ]
    sns.pointplot(
        data=facet_data,
        y="Temperature",
        x="Score",
        hue="Sampler",
        hue_order=["Basic", "Top-p", "Min-p"],
        palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
        order=[3.0, 2.0, 1.0],
        ax=ax,
        # Pointplot parameters to control appearance and avoid redundancy
        dodge=True,  # Dodge points along the categorical axis like the stripplot
        join=False,  # Do not draw lines connecting points across categories
        markers=["d", "o", "s"],  # Optional: different markers per hue level
        scale=0.7,  # Optional: adjust point size
        ci="sd",  # Show standard deviation, can be 95 for 95% CI, etc.
        errwidth=1,  # Width of error bar lines
        capsize=0.2,  # Width of the caps on error bars
    )
    # Remove the legend created by pointplot on each subplot to avoid duplicates
    # We'll keep the main legend created by catplot
    handles, labels = ax.get_legend_handles_labels()
    # Only keep the handles/labels corresponding to the hue levels for the legend
    # The pointplot adds extra handles we don't need in the main legend.
    # Assuming 3 hue levels, pointplot might add 3 handles for points + 3 for error bars
    # Adjust slice if number of hue levels or pointplot elements changes
    if handles:  # Check if handles exist before trying to remove legend
        ax.legend_.remove()


g.set(xlim=(0.5, 10.5))
# By default, y axis goes from top to bottom: 1.0, 2.0, 3.0.
# We want 3.0 on top and 1.0 on bottom.
# Uncomment if you need to invert y-axis (already commented in original)
# for ax in g.axes.flat:
#     ax.invert_yaxis()

# Move the original legend created by catplot
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

# Save the combined plot
# Make sure the filename reflects the new plot type
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=temp_x=score_hue=sampler_row=metric_col=diversity_stripnpoint",  # Updated filename
)
# plt.show()

# Create the base barplot using catplot
# Barplot shows mean and 95% CI by default
plt.close()
g = sns.catplot(
    data=filtered_data,
    kind="bar",  # Changed to "bar"
    y="Temperature",
    x="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    row="Metric",
    col="Diversity",
    margin_titles=True,
    order=[3.0, 2.0, 1.0],
    # Optional: Adjust bar plot parameters
    # ci=95, # Default confidence interval is 95%
    # errwidth=1.5, # Width of error bar lines
    # capsize=.1 # Width of the caps on error bars
    # alpha=0.7 # Make bars slightly transparent?
)

# Overlay the stripplot on each facet's axes
# Iterate through the axes dictionary provided by catplot
for (row_val, col_val), ax in g.axes_dict.items():
    # Filter data for the specific facet
    facet_data = filtered_data[
        (filtered_data["Metric"] == row_val) & (filtered_data["Diversity"] == col_val)
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

# Save the combined plot
# Make sure the filename reflects the new plot type
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=temp_x=score_hue=sampler_row=metric_col=diversity_barnstrip",  # Updated filename
)
plt.show()


# Create the base barplot using catplot
# Barplot shows mean and 95% CI by default
plt.close()
g = sns.catplot(
    data=filtered_data,
    kind="bar",  # Changed to "bar"
    y="Temperature",
    x="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    row="Diversity",
    col="Metric",
    margin_titles=True,
    order=[3.0, 2.0, 1.0],
    # Optional: Adjust bar plot parameters
    # ci=95, # Default confidence interval is 95%
    # errwidth=1.5, # Width of error bar lines
    # capsize=.1 # Width of the caps on error bars
    # alpha=0.7 # Make bars slightly transparent?
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

# Save the combined plot
# Make sure the filename reflects the new plot type
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=temp_x=score_hue=sampler_row=diversity_col=metric_barnstrip",  # Updated filename
)
plt.show()


# Create the base barplot using catplot
# Barplot shows mean and 95% CI by default
plt.close()
g = sns.catplot(
    data=filtered_data,
    kind="bar",  # Changed to "bar"
    y="Temperature",
    x="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    row="Diversity",
    col="Metric",
    margin_titles=True,
    order=[3.0, 2.0, 1.0],
    # Optional: Adjust bar plot parameters
    # ci=95, # Default confidence interval is 95%
    # errwidth=1.5, # Width of error bar lines
    # capsize=.1 # Width of the caps on error bars
    # alpha=0.7 # Make bars slightly transparent?
)

# Overlay the stripplot on each facet's axes
# Iterate through the axes dictionary provided by catplot
for (row_val, col_val), ax in g.axes_dict.items():
    # Filter data for the specific facet
    facet_data = filtered_data[
        (filtered_data["Metric"] == row_val) & (filtered_data["Diversity"] == col_val)
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

# Save the combined plot
# Make sure the filename reflects the new plot type
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=temp_x=score_hue=sampler_row=diversity_col=metric_barnstrip",  # Updated filename
)
# plt.show()


plt.close()
g = sns.catplot(
    data=raw_human_evals_scores_tall_df[
        raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
        & (raw_human_evals_scores_tall_df["Diversity"] == "High")
    ],
    kind="box",
    y="Temperature",
    x="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    row="Metric",
    col="Diversity",
    margin_titles=True,
    order=[3.0, 2.0, 1.0],
)
g.set(xlim=(0.5, 10.5))
# By default, y axis goes from top to bottom: 1.0, 2.0, 3.0.
# We want 3.0 on top and 1.0 on bottom.
# for ax in g.axes.flat:
#     ax.invert_yaxis()
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=temp_x=score_hue=sampler_row=metric_col=diversity_boxnwhisker",
)
# plt.show()


plt.close()
g = sns.catplot(
    data=raw_human_evals_scores_tall_df[
        raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
        & (raw_human_evals_scores_tall_df["Diversity"] == "High")
    ],
    kind="point",
    x="Temperature",
    y="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    col="Metric",
    row="Diversity",
    margin_titles=True,
    # order=[3.0, 2.0, 1.0],
)
g.set(ylim=(1, 10))
# By default, y axis goes from top to bottom: 1.0, 2.0, 3.0.
# We want 3.0 on top and 1.0 on bottom.
# for ax in g.axes.flat:
#     ax.invert_yaxis()
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=score_x=temp_hue=sampler_row=diversity_col=metric_point",
)
plt.show()


plt.close()
g = sns.catplot(
    data=raw_human_evals_scores_tall_df[
        raw_human_evals_scores_tall_df["Annotator Passed Attention Check"]
    ],
    kind="violin",
    y="Temperature",
    x="Score",
    hue="Sampler",
    hue_order=["Basic", "Top-p", "Min-p"],
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    col="Metric",
    row="Diversity",
    margin_titles=True,
    # cut=(),
)
g.set(xlim=(1, 10))
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="attentive_y=temp_x=score_hue=sampler_row=diversity_col=metric_violin",
)
# plt.show()

print()

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
    col_order=["High"],  # Only plot "High". This is intentional.
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

print("Finished notebooks/11_min_p_human_evals_raw")
