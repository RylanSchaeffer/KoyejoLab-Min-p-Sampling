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


def download_wandb_project_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: Optional[str] = None,
    filetype: str = "csv",
    max_workers: int = 10,  # New parameter to control the number of parallel workers
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


def download_wandb_project_runs_configs_helper(run):
    try:
        summary = run.summary._json_dict
        summary.update({k: v for k, v in run.config.items() if not k.startswith("_")})
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
        print(f"Error processing run {run.id}: {str(e)}")
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
