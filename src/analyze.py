import ast
import concurrent.futures
import hashlib
import numpy as np
import os
import pandas as pd
import pyarrow
import requests
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import wandb
from tqdm import tqdm

import src.globals


def compute_best_of_n_avg_scores_df(
    runs_scores_df: pd.DataFrame,
    Ns_list: Optional[List[int]] = None,
    num_repeats: int = 31,
) -> pd.DataFrame:
    if Ns_list is None:
        Ns_list = [1, 3, 6, 10, 31, 56, 100]

    # Step 1: Average scores over seeds.
    runs_avg_scores_df = (
        runs_scores_df.groupby(
            [
                "Model",
                "model_hf_path",
                "Model Type",
                "num_fewshot",
                "Sampler",
                "Sampler Value",
                "Task",
                "Temperature",
            ]
        )
        .agg(
            {
                "Exact Match (Strict)": "mean",
                "Exact Match (Flexible)": "mean",
            }
        )
        .reset_index()
    )

    best_of_n_avg_scores_dfs_list = []
    for (
        model,
        model_hf_path,
        model_type,
        num_fewshot,
        sampler,
        task,
    ), subset_df in runs_avg_scores_df.groupby(
        ["Model", "model_hf_path", "Model Type", "num_fewshot", "Sampler", "Task"]
    ):
        # Step 2: Repeat many times.
        for repeat_idx in range(num_repeats):
            # For each value of N.
            for N in Ns_list:
                try:
                    # Choose a subset of size N.
                    subset_of_size_N_df = subset_df.sample(n=N, replace=False)
                except ValueError:
                    # If the subset is smaller than N, skip it.
                    continue
                # Take the max score of the two metrics.
                best_exact_match_strict = subset_of_size_N_df[
                    "Exact Match (Strict)"
                ].max()
                best_exact_match_flexible = subset_of_size_N_df[
                    "Exact Match (Flexible)"
                ].max()
                best_of_n_avg_score_df = pd.DataFrame(
                    {
                        "Model": [model],
                        "model_hf_path": [model_hf_path],
                        "Model Type": [model_type],
                        "num_fewshot": [num_fewshot],
                        "Sampler": [sampler],
                        "Task": [task],
                        "repeat_idx": [repeat_idx],
                        "Number of Hyperparameters Swept": [N],
                        "Exact Match (Strict)": [best_exact_match_strict],
                        "Exact Match (Flexible)": [best_exact_match_flexible],
                    }
                )
                best_of_n_avg_scores_dfs_list.append(best_of_n_avg_score_df)

    best_of_n_avg_scores_df = pd.concat(
        best_of_n_avg_scores_dfs_list, ignore_index=True
    ).reset_index(drop=True)
    return best_of_n_avg_scores_df


def compute_diff_of_best_of_n_avg_scores_df(
    runs_scores_df: pd.DataFrame,
    Ns_list: Optional[List[int]] = None,
    num_repeats: int = 31,
) -> pd.DataFrame:
    if Ns_list is None:
        Ns_list = [1, 3, 6, 10, 31, 56, 100]

    # Step 1: Average scores over seeds.
    runs_avg_scores_df = (
        runs_scores_df.groupby(
            [
                "Model",
                "model_hf_path",
                "Model Type",
                "num_fewshot",
                "Sampler",
                "Sampler Value",
                "Task",
                "Temperature",
            ]
        )
        .agg(
            {
                "Exact Match (Strict)": "mean",
                "Exact Match (Flexible)": "mean",
            }
        )
        .reset_index()
    )

    def subsample_df_at_most_N(
        df: pd.DataFrame,
        N: int,
    ) -> pd.DataFrame:
        return df.sample(n=min(N, len(df)), replace=False)

    diff_of_best_of_n_avg_scores_dfs_list = []
    for (
        model,
        model_hf_path,
        model_type,
        num_fewshot,
        task,
    ), subset_df in runs_avg_scores_df.groupby(
        ["Model", "model_hf_path", "Model Type", "num_fewshot", "Task"]
    ):
        # Step 2: Repeat many times.
        for repeat_idx in range(num_repeats):
            # For each value of N.
            for N in Ns_list:
                # Choose a subset of size N for each sampler.
                # Note: "Basic" sampling doesn't have more than 31 hyperparameters, so we take
                # at most N per sampler.
                subset_of_at_most_N_per_sampler_df = subset_df.groupby("Sampler").apply(
                    subsample_df_at_most_N, N=N
                )

                assert len(subset_of_at_most_N_per_sampler_df)

                # Identify which rows correspond to Min-p.
                min_p_rows = subset_of_at_most_N_per_sampler_df["Sampler"] == "Min-p"

                # Take the max exact match (strict) score of Min-p.
                min_p_best_exact_match_strict = subset_of_at_most_N_per_sampler_df[
                    min_p_rows
                ]["Exact Match (Strict)"].max()

                # Take the max exact match (flexible) score of Min-p.
                min_p_best_exact_match_flexible = subset_of_at_most_N_per_sampler_df[
                    min_p_rows
                ]["Exact Match (Flexible)"].max()

                # Take the max exact match (strict) score of non-Min-p samplers.
                non_min_p_best_exact_match_strict = subset_of_at_most_N_per_sampler_df[
                    ~min_p_rows
                ]["Exact Match (Strict)"].max()

                # Take the max exact match (flexible) score of non-Min-p samplers.
                non_min_p_best_exact_match_flexible = (
                    subset_of_at_most_N_per_sampler_df[~min_p_rows][
                        "Exact Match (Flexible)"
                    ].max()
                )

                if np.isnan(min_p_best_exact_match_strict) or np.isnan(
                    non_min_p_best_exact_match_strict
                ):
                    raise ValueError("Something is not correct...")

                diff_of_best_of_n_avg_score_df = pd.DataFrame(
                    {
                        "Model": [model],
                        "model_hf_path": [model_hf_path],
                        "Model Type": [model_type],
                        "num_fewshot": [num_fewshot],
                        "Task": [task],
                        "repeat_idx": [repeat_idx],
                        "Number of Hyperparameters Swept": [N],
                        "Best Min-p Exact Match - Best Other Exact Match (Strict)": [
                            min_p_best_exact_match_strict
                            - non_min_p_best_exact_match_strict
                        ],
                        "Best Min-p Exact Match - Best Other Exact Match (Flexible)": [
                            min_p_best_exact_match_flexible
                            - non_min_p_best_exact_match_flexible
                        ],
                    }
                )
                diff_of_best_of_n_avg_scores_dfs_list.append(
                    diff_of_best_of_n_avg_score_df
                )

    diff_of_best_of_n_avg_scores_df = pd.concat(
        diff_of_best_of_n_avg_scores_dfs_list, ignore_index=True
    ).reset_index(drop=True)
    return diff_of_best_of_n_avg_scores_df


def compute_samplers_pairwise_scores_differences_df(
    runs_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    samplers = runs_scores_df["Sampler"].unique()

    sampler_pairwise_differences_dfs_list = []
    for (
        model,
        model_hf_path,
        num_fewshot,
        task,
    ), subset_df in runs_scores_df.groupby(
        ["Model", "model_hf_path", "num_fewshot", "Task"]
    ):
        # for sampler1_idx, sampler1 in enumerate(samplers):
        sampler1 = "Min-p"
        for sampler2 in samplers:
            if sampler2 == sampler1:
                continue
            sampler1_scores = subset_df[subset_df["Sampler"] == sampler1][
                "Exact Match (Strict)"
            ].values
            sampler2_scores = subset_df[subset_df["Sampler"] == sampler2][
                "Exact Match (Strict)"
            ].values

            # Compute the distribution of pairwise differences of scores.
            # Shape: (sampler1 samples, sampler2 samples)
            pairwise_differences = sampler1_scores[:, None] - sampler2_scores
            pairwise_differences_df = pd.DataFrame(
                pairwise_differences.flatten(),
                columns=["Difference of Exact Matches (Strict)"],
            )
            pairwise_differences_df["Model"] = model
            pairwise_differences_df["model_hf_path"] = model_hf_path
            pairwise_differences_df["num_fewshot"] = num_fewshot
            pairwise_differences_df["Task"] = task
            pairwise_differences_df["Sampler1"] = sampler1
            pairwise_differences_df["Sampler2"] = sampler2
            pairwise_differences_df["Sampler1 - Sampler2"] = f"{sampler1} - {sampler2}"

            sampler_pairwise_differences_dfs_list.append(pairwise_differences_df)

    sampler_pairwise_differences_df = pd.concat(
        sampler_pairwise_differences_dfs_list, ignore_index=True
    ).reset_index(drop=True)
    return sampler_pairwise_differences_df


def download_wandb_project_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: Optional[str] = None,
    filetype: str = "csv",
    max_workers: int = 30,  # New parameter to control the number of parallel workers
) -> pd.DataFrame:
    assert filetype in {"csv", "feather", "parquet"}

    filename = "sweeps=" + ",".join(sweep_ids)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    runs_configs_df_path = os.path.join(
        data_dir, hashed_filename + f"_runs_configs.{filetype}"
    )

    if refresh or not os.path.isfile(runs_configs_df_path):
        print(f"Creating {runs_configs_df_path} anew.")

        api = wandb.Api(timeout=600)

        if wandb_username is None:
            wandb_username = api.viewer.username

        sweep_results_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_run = {}

            for sweep_id in sweep_ids:
                try:
                    sweep = api.sweep(
                        f"{wandb_username}/{wandb_project_path}/{sweep_id}"
                    )
                    for run in sweep.runs:
                        future_to_run[
                            executor.submit(
                                download_wandb_project_runs_configs_helper, run
                            )
                        ] = run
                except Exception as e:
                    print(f"Error processing sweep {sweep_id}: {str(e)}")

            for future in tqdm(
                concurrent.futures.as_completed(future_to_run), total=len(future_to_run)
            ):
                result = future.result()
                if result is not None:
                    sweep_results_list.append(result)

        runs_configs_df = pd.DataFrame(sweep_results_list)
        runs_configs_df.reset_index(inplace=True, drop=True)

        # For some unknown reason (maybe my error handling?), sometimes, the API returns multiple
        # copies of the same run_id. Let's drop duplicates based on the run_id.
        runs_configs_df.drop_duplicates(subset=["run_id"], inplace=True)

        # Tidy up variable names.
        runs_configs_df["Model"] = runs_configs_df["model_hf_path"].map(
            src.globals.MODELS_NICE_NAMES_DICT
        )
        runs_configs_df["Model Type"] = runs_configs_df["model_hf_path"].map(
            src.globals.MODELS_TYPE_DICT
        )
        runs_configs_df["Sampler"] = runs_configs_df["sampler"].map(
            src.globals.SAMPLERS_NICE_NAMES_DICT
        )
        runs_configs_df["Task"] = runs_configs_df["task"].map(
            src.globals.TASK_NICE_NAMES_DICT
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

        # Save to disk
        runs_configs_df.to_csv(
            runs_configs_df_path.replace(filetype, "csv"), index=False
        )
        try:
            runs_configs_df.to_feather(
                runs_configs_df_path.replace(filetype, "feather")
            )
        except Exception as e:
            print(f"Error saving to feather: {str(e)}")

        try:
            runs_configs_df.to_parquet(
                runs_configs_df_path.replace(filetype, "parquet"), index=False
            )
        except Exception as e:
            print(f"Error saving to parquet: {str(e)}")

        print(f"Regenerated and wrote {runs_configs_df_path} to disk.")
        del runs_configs_df

    print(f"Reading {runs_configs_df_path} from disk.")
    if filetype == "csv":
        runs_configs_df = pd.read_csv(runs_configs_df_path)
    elif filetype == "feather":
        runs_configs_df = pd.read_feather(runs_configs_df_path)
    elif filetype == "parquet":
        runs_configs_df = pd.read_parquet(runs_configs_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
    print(f"Loaded {runs_configs_df_path} from disk.")

    # Keep only finished runs
    finished_runs = runs_configs_df["State"] == "finished"
    print(
        f"% of successfully finished runs: {100.0 * finished_runs.mean():.2f}% ({finished_runs.sum()} / {len(finished_runs)})"
    )

    if finished_only:
        runs_configs_df = runs_configs_df[finished_runs]

        # Check that we don't have an empty data frame.
        assert len(runs_configs_df) > 0

        # Ensure we aren't working with a slice.
        runs_configs_df = runs_configs_df.copy()

    return runs_configs_df


def download_wandb_project_runs_configs_helper(run, num_attempts: int = 5):
    for attempt_idx in range(num_attempts):
        try:
            summary = run.summary._json_dict
            summary.update(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )
            summary.update(
                {
                    "State": run.state,
                    "Sweep": run.sweep.id if run.sweep is not None else None,
                    "run_id": run.id,
                    "run_name": run.name,
                }
            )
            return summary
        except Exception as e:
            print(f"Error processing run {run.id} on attempt {attempt_idx}: {str(e)}")
            print(e)
            time.sleep(30)
    print(f"Unable to process run {run.id}")
    return None


def setup_notebook_dir(
    notebook_dir: str,
    refresh: bool = False,
) -> Tuple[str, str]:
    # Declare paths.
    data_dir = os.path.join(notebook_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    results_dir = os.path.join(notebook_dir, "results")
    if refresh:
        import shutil

        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir
