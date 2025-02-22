import matplotlib.pyplot as plt
import matplotlib.transforms
import os
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
    "bx2lxry7",  # GSM8K CoT Min-P
]

runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="min-p-evals",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
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


plt.close()
g = sns.relplot(
    data=runs_configs_df[runs_configs_df["task"] == "gsm8k_cot_llama"],
    kind="scatter",
    x="sampler_value",
    y="exact_match_strict-match",
    hue="temperature",
    style="Model Type",
    style_order=src.globals.MODELS_TYPE_ORDER_LIST,
    palette="coolwarm",
    col="Model",
    col_order=src.globals.MODELS_ORDER_LIST,
    row="Sampler",
    row_order=src.globals.SAMPLERS_ORDER_LIST,
    facet_kws={"sharex": "row", "sharey": True, "margin_titles": True},
)
g.set(xlabel="Sampler Value", ylabel="Exact Match (Strict)")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
plt.show()

# plt.close()
# g = sns.relplot(
#     data=runs_configs_df,
#     x="sampler_value",
#     y="exact_match_flexible-extract",
#     hue="temperature",
#     col="model_hf_path",
#     col_order=src.globals.MODELS_ORDER_LIST,
#     row="task",
#     row_order=src.globals.TASKS_ORDER_LIST,
#     facet_kws={"sharey": "row"},
# )
# g.set(xlabel="Sampler Value", ylabel="Exact Match (Strict)")
# g.set_titles("{col_name}", "{row_name}")
# plt.show()
