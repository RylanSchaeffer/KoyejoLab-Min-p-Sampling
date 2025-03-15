import matplotlib.pyplot as plt
import matplotlib.transforms
import os

import numpy as np
import pandas as pd
import seaborn as sns
import wandb

import src.analyze
import src.globals
import src.plot


refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

wandb_sweep_ids = [
    "fco8drwz",  # GSM8K CoT Llama Basic Part 1.
    "8mzf5s01",  # GSM8K CoT Llama Basic Part 2.
    "23dj4sc5",  # GSM8K CoT Llama Top-p Part 1.
    "wutvcxa6",  # GSM8K CoT Llama Top-p Part 2.
    "k1k9o4o3",  # GSM8K CoT Llama Top-k Part 1.
    "wav3hizz",  # GSM8K CoT Llama Top-k Part 2.
    "bx2lxry7",  # GSM8K CoT Llama Min-p Part 1.
    "qdrjunpa",  # GSM8K CoT Llama Min-p Part 2.
]

runs_scores_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="min-p-evals",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username="rylan",
    # wandb_username=wandb.api.default_entity,
    finished_only=True,
    max_workers=30,
)

num_repeats = 150

# TODO: Debug why W&B API does not grab 100% of runs; seems to grab >99% but not 100%.
# See GitHub issue: https://github.com/wandb/wandb/issues/8594
# for cols, subset_df in runs_scores_df.groupby(
#     [
#         "Model",
#         "model_hf_path",
#         "Model Type",
#         "num_fewshot",
#         "Sampler",
#         "Sampler Value",
#         "Task",
#         "Temperature",
#     ]
# ):
#     if len(subset_df) < 3:
#         print(cols)
#         print(len(subset_df))

Ns_list = np.unique(
    np.logspace(0, np.log10(100), 60).astype(
        int
    )  # We have max 180 hyperparameters per sampler.
).tolist()

diff_of_best_of_n_avg_scores_df = src.analyze.compute_diff_of_best_of_n_avg_scores_df(
    runs_scores_df,
    Ns_list=Ns_list,
    num_repeats=num_repeats,
)
plt.close()
g = sns.relplot(
    data=diff_of_best_of_n_avg_scores_df,
    kind="line",
    x="Number of Hyperparameters Swept",
    y="Best Min-p Exact Match - Best Other Exact Match (Strict)",
    style="Model Type",
    style_order=src.globals.MODELS_TYPE_ORDER_LIST,
    # palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    col="Model",
    col_order=src.globals.MODELS_ORDER_LIST,
    col_wrap=4,
    facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
)
# Add dashed horizontal line at 0.
for ax in g.axes.flat:
    ax.axhline(0, color="black", linestyle="--")
g.set(ylabel="Best Min-p - Best Other Sampler", ylim=(-0.2, 0.1))
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "center right", bbox_to_anchor=(0.95, 0.1), frameon=True)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=diff_of_em_strict_x=N_hue=sampler_row=model",
)
g.set(
    xscale="log",
    # xlabel="Number of Hyperparameters Swept",
    # ylabel="Best Exact Match (Strict)",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=diff_of_em_strict_x=log_N_hue=sampler_row=model",
)
# plt.show()


best_of_n_avg_scores_df = src.analyze.compute_best_of_n_avg_scores_df(
    runs_scores_df,
    Ns_list=Ns_list,
    num_repeats=num_repeats,
)
plt.close()
g = sns.relplot(
    data=best_of_n_avg_scores_df,
    kind="line",
    x="Number of Hyperparameters Swept",
    y="Exact Match (Strict)",
    hue="Sampler",
    hue_order=src.globals.SAMPLERS_ORDER_LIST,
    style="Model Type",
    style_order=src.globals.MODELS_TYPE_ORDER_LIST,
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    col="Model",
    col_order=src.globals.MODELS_ORDER_LIST,
    col_wrap=4,
    facet_kws={"margin_titles": True, "sharey": False, "sharex": True},
)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "center right", bbox_to_anchor=(0.95, 0.1), frameon=True)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=em_strict_x=N_hue=sampler_row=model",
)
g.set(
    xscale="log",
    # xlabel="Number of Hyperparameters Swept",
    # ylabel="Best Exact Match (Strict)",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=em_strict_x=log_N_hue=sampler_row=model",
)
# plt.show()

# Average over repeat index for Best of N, then plot scaling behavior.
best_of_n_avg_scores_over_repeats_df = (
    best_of_n_avg_scores_df[
        ["Sampler", "Model", "Number of Hyperparameters Swept", "Exact Match (Strict)"]
    ]
    .groupby(["Sampler", "Model", "Number of Hyperparameters Swept"])[
        "Exact Match (Strict)"
    ]
    .mean()
    .reset_index()
)
best_of_n_avg_scores_over_repeats_df["Neg Log Exact Match (Strict)"] = -np.log(
    best_of_n_avg_scores_over_repeats_df["Exact Match (Strict)"]
)

plt.close()
g = sns.relplot(
    data=best_of_n_avg_scores_over_repeats_df,
    kind="line",
    x="Number of Hyperparameters Swept",
    y="Neg Log Exact Match (Strict)",
    hue="Sampler",
    hue_order=src.globals.SAMPLERS_ORDER_LIST,
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    col="Model",
    col_order=src.globals.MODELS_ORDER_LIST,
    col_wrap=6,
    # row="Task",
    # row_order=src.globals.TASKS_ORDER_LIST,
    facet_kws={"margin_titles": True, "sharey": False, "sharex": True},
)
g.set(
    xscale="log",
    yscale="log",
)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_em_strict_x=N_hue=sampler_row=model",
)
# plt.show()

plt.close()
g = sns.displot(
    data=runs_scores_df,
    kind="kde",
    x="Exact Match (Strict)",
    hue="Sampler",
    hue_order=src.globals.SAMPLERS_ORDER_LIST,
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    # row="Task",
    # row_order=src.globals.TASKS_ORDER_LIST,
    col="Model",
    col_order=src.globals.MODELS_ORDER_LIST,
    col_wrap=6,
    facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
    common_norm=False,  # Normalize each hue separately
    clip=(0, 1),  # Clip the density to [0, 1] because exact match.
)
g.set(xlim=(0, 1))
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=kde_x=em_strict_hue=sampler_row=model",
)
# plt.show()

plt.close()
g = sns.displot(
    data=runs_scores_df,
    kind="kde",
    x="Exact Match (Flexible)",
    hue="Sampler",
    hue_order=src.globals.SAMPLERS_ORDER_LIST,
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    # row="Task",
    # row_order=src.globals.TASKS_ORDER_LIST,
    col="Model",
    col_order=src.globals.MODELS_ORDER_LIST,
    col_wrap=6,
    facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
    common_norm=False,  # Normalize each hue separately
    clip=(0, 1),  # Clip the density to [0, 1] because exact match.
)
g.set(xlim=(0, 1))
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=kde_x=em_flexible_hue=sampler_row=model",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=runs_scores_df,
    kind="line",
    x="Sampler Value",
    y="Exact Match (Strict)",
    hue="Temperature",
    palette="coolwarm",
    style="Model Type",
    style_order=src.globals.MODELS_TYPE_ORDER_LIST,
    row="Model",
    row_order=src.globals.MODELS_ORDER_LIST,
    col="Sampler",
    col_order=src.globals.SAMPLERS_ORDER_LIST,
    facet_kws={"sharex": "col", "sharey": "row", "margin_titles": True},
)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=em_strict_x=sampler_value_hue=temperature_style=model_type_row=model_col=sampler",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=runs_scores_df,
    kind="line",
    x="Sampler Value",
    y="Exact Match (Flexible)",
    hue="Temperature",
    style="Model Type",
    style_order=src.globals.MODELS_TYPE_ORDER_LIST,
    palette="coolwarm",
    row="Model",
    row_order=src.globals.MODELS_ORDER_LIST,
    col="Sampler",
    col_order=src.globals.SAMPLERS_ORDER_LIST,
    facet_kws={"sharex": "col", "sharey": "row", "margin_titles": True},
)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=em_flexible_x=sampler_value_hue=temperature_style=model_type_row=model_col=sampler",
)

# plt.show()


samplers_pairwise_scores_differences_df: pd.DataFrame = (
    src.analyze.compute_samplers_pairwise_scores_differences_df(
        runs_scores_df=runs_scores_df,
    )
)

plt.close()
g = sns.displot(
    data=samplers_pairwise_scores_differences_df,
    kind="kde",
    x="Difference of Exact Matches (Strict)",
    hue="Sampler1 - Sampler2",
    # row="Task",
    # row_order=src.globals.TASKS_ORDER_LIST,
    col="Model",
    col_order=src.globals.MODELS_ORDER_LIST,
    col_wrap=6,
    facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
    common_norm=False,  # Normalize each hue separately
)
g.set(xlabel="Sampler 1's Exact Match - Sampler 2's Exact Match")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=kde_x=sampler_pairwise_diff_hue=sampler1_sampler2_row=model",
)
# plt.show()


plt.close()
g = sns.displot(
    data=samplers_pairwise_scores_differences_df,
    kind="ecdf",
    complementary=True,
    x="Difference of Exact Matches (Strict)",
    hue="Sampler1 - Sampler2",
    row="Task",
    row_order=src.globals.TASKS_ORDER_LIST,
    col="Model",
    col_order=src.globals.MODELS_ORDER_LIST,
    facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
)
g.set(
    xlabel="Sampler 1's Exact Match - Sampler 2's Exact Match",
    ylabel="1 - Empirical CDF",
)
for ax in g.axes.flat:
    ax.axhline(0.5, color="black", linestyle="--")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=survival_x=sampler_pairwise_diff_hue=sampler1_sampler2_row=model",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=runs_scores_df,
    kind="line",
    x="Temperature",
    y="_runtime",
    hue="Sampler",
    hue_order=src.globals.SAMPLERS_ORDER_LIST,
    style="Model Type",
    style_order=src.globals.MODELS_TYPE_ORDER_LIST,
    col="Model",
    col_order=src.globals.MODELS_ORDER_LIST,
    col_wrap=6,
    facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
)
g.set(
    ylabel="Runtime (s)",
)
g.set_titles(col_template="{col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=runtime_x=temperature_hue=sampler_style=type_col=model",
)
plt.show()

print("Finished notebooks/00_gsm8k_cot_llama")
