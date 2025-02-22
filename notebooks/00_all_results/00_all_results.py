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


# refresh = False
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

wandb_sweep_ids = [
    "23dj4sc5",  # GSM8K CoT Top-P
    "k1k9o4o3",  # GSM8K CoT Top-K
    "bx2lxry7",  # GSM8K CoT Min-P
]

runs_scores_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="min-p-evals",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username="rylan",
    # wandb_username=wandb.api.default_entity,
    finished_only=True,
    max_workers=60,
)


best_of_n_avg_scores_df = src.analyze.compute_best_of_n_scores(
    runs_scores_df,
    num_repeats=100,
    Ns_list=np.unique(
        np.logspace(0, np.log10(181), 40).astype(
            int
        )  # We have max 180 hyperparameters per sampler.
    ).tolist(),
)


plt.close()
g = sns.relplot(
    data=best_of_n_avg_scores_df,
    kind="line",
    x="N",
    y="Exact Match (Strict)",
    hue="Sampler",
    hue_order=src.globals.SAMPLERS_ORDER_LIST,
    palette=sns.hls_palette(len(src.globals.SAMPLERS_ORDER_LIST)),
    row="Model",
    # row_order=src.globals.MODELS_ORDER_LIST,
    col="Task",
    # col_order=src.globals.TASKS_ORDER_LIST,
    facet_kws={"margin_titles": True, "sharey": "row"},
)
g.set(
    xscale="log",
    xlabel="Number of Hyperparameters Swept",
    ylabel="Best Exact Match (Strict)",
)
g.set_titles(col_template="Task: {col_name}", row_template="{row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=em_strict_x=N_hue=sampler_row=model_col=task",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=runs_scores_df[runs_scores_df["Task"] == "GSM8K CoT"],
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
    plot_filename="task=gsm8k_y=em_strict_x=sampler_value_hue=temperature_style=model_type_row=model_col=sampler",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=runs_scores_df[runs_scores_df["Task"] == "GSM8K CoT"],
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
    plot_filename="task=gsm8k_y=em_flexible_x=sampler_value_hue=temperature_style=model_type_row=model_col=sampler",
)

# plt.show()

print("Finished notebooks/00_all_results")
