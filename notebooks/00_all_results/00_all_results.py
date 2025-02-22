import matplotlib.pyplot as plt
import matplotlib.transforms
import os
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
    "23dj4sc5",  # GSM8K CoT Top-P
    "bx2lxry7",  # GSM8K CoT Min-P
]

runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="min-p-evals",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username="rylan",
    # wandb_username=wandb.api.default_entity,
    finished_only=True,
)
runs_configs_df["Model"] = runs_configs_df["model_hf_path"].map(
    src.globals.MODELS_NICE_NAMES_DICT
)
runs_configs_df["Model Type"] = runs_configs_df["model_hf_path"].map(
    src.globals.MODELS_TYPE_DICT
)
runs_configs_df["Sampler"] = runs_configs_df["sampler"].map(
    src.globals.SAMPLERS_NICE_NAMES_DICT
)
runs_configs_df.rename(
    columns={
        "temperature": "Temperature",
        "sampler_value": "Sampler Value",
        "exact_match_strict-match": "Exact Match (Strict)",
        "exact_match_flexible-extract": "Exact Match (Flexible)",
    },
    inplace=True,
)


plt.close()
g = sns.relplot(
    data=runs_configs_df[runs_configs_df["task"] == "gsm8k_cot_llama"],
    kind="line",
    x="Sampler Value",
    y="Exact Match (Strict)",
    hue="Temperature",
    style="Model Type",
    style_order=src.globals.MODELS_TYPE_ORDER_LIST,
    palette="coolwarm",
    row="Model",
    row_order=src.globals.MODELS_ORDER_LIST,
    col="Sampler",
    col_order=src.globals.SAMPLERS_ORDER_LIST,
    facet_kws={"sharex": "col", "sharey": "row", "margin_titles": True},
    # s=50,
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
    data=runs_configs_df[runs_configs_df["task"] == "gsm8k_cot_llama"],
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
    # s=50,
)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="task=gsm8k_y=em_flexible_x=sampler_value_hue=temperature_style=model_type_row=model_col=sampler",
)

# plt.show()

print("Finished notebooks/00_all_results")
