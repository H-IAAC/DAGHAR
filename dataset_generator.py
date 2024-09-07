from pathlib import Path
import os, shutil
import pandas as pd
from typing import Tuple, List, Dict
import argparse

from steps import (
    SplitGuaranteeingAllClassesPerSplit,
    BalanceToMinimumClass,
    BalanceToMinimumClassAndUser,
    FilterByCommonRows,
)

from pipelines import match_columns, pipelines

# Set the seed for reproducibility
import numpy as np
import random
import traceback

np.random.seed(42)
random.seed(42)
# pd.np.random.seed(42)

from readers import (
    read_kuhar,
    read_motionsense,
    read_wisdm,
    read_uci,
    read_realworld,
    sanity_function,
    real_world_organize,
)

"""This module is used to generate the datasets. The datasets are generated in the following steps:    
    1. Read the raw dataset
    2. Preprocess the raw dataset
    3. Preprocess the standardized dataset
    4. Remove activities that are equal to -1
    5. Balance the dataset per activity
    6. Balance the dataset per user and activity
    7. Save the datasets
    8. Generate the views of the datasets

    The datasets are generated in the following folders:
    1. data/unbalanced: The unbalanced dataset
    2. data/raw_balanced: The raw balanced dataset per activity
    3. data/standardized_balanced: The standardized balanced dataset per activity
    4. data/raw_balanced_user: The raw balanced dataset per user and activity (NOT USED)
    5. data/standardized_balanced_user: The standardized balanced dataset per user and activity (NOT USED)

    The datasets are generated in the following format:
    1. data/unbalanced/{dataset}/unbalanced.csv: The unbalanced dataset
    2. data/raw_balanced/{dataset}/train.csv: The raw balanced train dataset per activity
    3. data/raw_balanced/{dataset}/validation.csv: The raw balanced validation dataset per activity
    4. data/raw_balanced/{dataset}/test.csv: The raw balanced test dataset per activity
    5. data/standardized_balanced/{dataset}/train.csv: The standardized balanced train dataset per activity
    6. data/standardized_balanced/{dataset}/validation.csv: The standardized balanced validation dataset per activity
    7. data/standardized_balanced/{dataset}/test.csv: The standardized balanced test dataset per activity
    8. data/raw_balanced_user/{dataset}/train.csv: The raw balanced train dataset per user and activity (NOT USED)
    9. data/raw_balanced_user/{dataset}/validation.csv: The raw balanced validation dataset per user and activity (NOT USED)
    10. data/raw_balanced_user/{dataset}/test.csv: The raw balanced test dataset per user and activity (NOT USED)
    11. data/standardized_balanced_user/{dataset}/train.csv: The standardized balanced train dataset per user and activity (NOT USED)
    12. data/standardized_balanced_user/{dataset}/validation.csv: The standardized balanced validation dataset per user and activity (NOT USED)
    13. data/standardized_balanced_user/{dataset}/test.csv: The standardized balanced test dataset per user and activity (NOT USED)
"""

# Dictionary of dataset paths
dataset_paths: Dict[str, str] = {
    "KuHar": "KuHar/1.Raw_time_domian_data",
    "MotionSense": "MotionSense/A_DeviceMotion_data",
    "WISDM": "WISDM/wisdm-dataset/raw/phone",
    "UCI": "UCI/RawData",
    "RealWorld": "RealWorld/realworld2016_dataset",
}

# Dictionary with datasets and their respesctive reader functions
dataset_readers: Dict[str, callable] = {
    "KuHar": read_kuhar,
    "MotionSense": read_motionsense,
    "WISDM": read_wisdm,
    "UCI": read_uci,
    "RealWorld": read_realworld,
}

# Preprocess the datasets

# Path to save the datasets
output_path: Path = Path("data/datasets")

balancer_activity: object = BalanceToMinimumClass(
    class_column="standard activity code"
)
balancer_activity_and_user: object = BalanceToMinimumClassAndUser(
    class_column="standard activity code", filter_column="user"
)

split_data: object = SplitGuaranteeingAllClassesPerSplit(
    column_to_split="user",
    class_column="standard activity code",
    train_size=0.8,
    random_state=42,
)

split_data_train_val: object = SplitGuaranteeingAllClassesPerSplit(
    column_to_split="user",
    class_column="standard activity code",
    train_size=0.9,
    random_state=42,
)


def balance_per_activity(
    dataset: str, dataframe: pd.DataFrame, output_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """This function balance the dataset per activity and save the balanced dataset.

    Parameters
    ----------
    dataset : str
        The dataset name.
    dataframe : pd.DataFrame
        The dataset.
    output_path : str
        The path to save the balanced dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The first element is the train dataset, the second is the validation dataset and the third is the test dataset.
    """

    train_df, test_df = split_data(dataframe)
    train_df, val_df = split_data_train_val(train_df)

    train_df = balancer_activity(train_df)
    val_df = balancer_activity(val_df)
    test_df = balancer_activity(test_df)

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Raw data balanced per activity saved at {output_dir}")

    return train_df, val_df, test_df


def balance_per_user_and_activity(dataset, dataframe, output_path):
    """The function balance the dataset per user and activity and save the balanced dataset.

    Parameters
    ----------
    dataset : str
        The dataset name.
    dataframe : pd.DataFrame
        The dataset.
    output_path : str
        The path to save the balanced dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The first element is the train dataset, the second is the validation dataset and the third is the test dataset.
    """
    new_df_balanced = balancer_activity_and_user(
        dataframe[dataframe["standard activity code"] != -1]
    )
    train_df, test_df = split_data(new_df_balanced)
    train_df, val_df = split_data_train_val(train_df)

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Raw data balanced per user and activity saved at {output_dir}")

    return train_df, val_df, test_df


def generate_views(
    new_df,
    new_df_standardized,
    dataset,
    path_balanced,
    path_balanced_standardized,
):
    """This function generate the views of the dataset.

    Parameters
    ----------
    new_df : pd.DataFrame
        The raw dataset.
    new_df_standardized : pd.DataFrame
        The standardized dataset.
    dataset : str
        The dataset name.
    """

    # Filter the datasets by equal elements
    filter_common = FilterByCommonRows(match_columns=match_columns[dataset])
    new_df, new_df_standardized = filter_common(new_df, new_df_standardized)

    # Preprocess and save the raw balanced dataset per activity
    print(" ---- RAW")
    train_df, val_df, test_df = balance_per_activity(
        dataset, new_df, path_balanced
    )
    sanity_function(train_df, val_df, test_df)

    # Preprocess and save the standardized balanced dataset per activity
    print(" ---- STANDARDIZED")
    train_df, val_df, test_df = balance_per_activity(
        dataset, new_df_standardized, path_balanced_standardized
    )
    sanity_function(train_df, val_df, test_df)


def main(datasets_to_process: List[str], output_path: str):
    """This is the main function to generate the datasets. It will loop through
    and their respective pipelines to generate the datasets.

    Parameters
    ----------
    datasets_to_process : List[str]
        A list of datasets to process.
    output_path : str
        The path to save the datasets.
    """
    # Creating the datasets
    for dataset in datasets_to_process:
        print(f"Preprocessing the dataset {dataset} ...\n")

        reader = dataset_readers[dataset]

        # Read the raw dataset
        if dataset == "RealWorld":
            print("Organizing the RealWorld dataset ...\n")
            # Create a folder to save the organized dataset
            workspace = Path(
                "data/original/RealWorld/realworld2016_dataset_organized"
            )
            if not os.path.isdir(workspace):
                os.mkdir(workspace)
            # Organize the dataset
            workspace, users = real_world_organize()
            path = workspace
            raw_dataset = reader(path, users)
            # Preprocess the raw dataset
            print(f"Preprocess the raw dataset: {dataset}\n")
            new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
            for view_name in pipelines[dataset].keys():
                try:
                    if "standardized" in view_name:
                        print(
                            f"Preprocess the dataset {dataset} with {view_name} ...\n"
                        )

                        # Preprocess the standardized dataset
                        new_df_standardized = pipelines[dataset][view_name](
                            raw_dataset
                        )
                        # Remove activities that are equal to -1
                        new_df = new_df[new_df["standard activity code"] != -1]
                        new_df_standardized = new_df_standardized[
                            new_df_standardized["standard activity code"] != -1
                        ]
                        generate_views(
                            new_df,
                            new_df_standardized,
                            dataset,
                            path_balanced=output_path / f"raw_balanced",
                            path_balanced_standardized=output_path / f"{view_name}",
                        )
                        positions = new_df["position"].unique()
                        for position in list(positions):
                            new_df_filtered = new_df[
                                new_df["position"] == position
                            ]
                            new_df_standardized_filtered = new_df_standardized[
                                new_df_standardized["position"] == position
                            ]
                            new_dataset = dataset + "_" + position
                            generate_views(
                                new_df_filtered,
                                new_df_standardized_filtered,
                                new_dataset,
                                path_balanced=output_path  / f"{view_name}_balanced",
                                path_balanced_standardized=output_path / f"{view_name}_balanced_standardized",
                            )
                except Exception as e:
                    print(
                        f"Error generating the view {view_name} for {dataset}: {e}"
                    )
                    traceback.print_exc()
                    continue
        else:
            path = Path(f"data/original/{dataset_paths[dataset]}")
            raw_dataset = reader(path)
            # Preprocess the raw dataset
            print(f"Preprocess the raw dataset {dataset}\n")
            new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
            for view_name in pipelines[dataset].keys():
                try:
                    if "standardized" in view_name:
                        print(
                            f"Preprocess the dataset {dataset} with {view_name} ...\n"
                        )

                        # Preprocess the standardized dataset
                        new_df_standardized = pipelines[dataset][view_name](
                            raw_dataset
                        )
                        # Remove activities that are equal to -1
                        new_df = new_df[new_df["standard activity code"] != -1]
                        new_df_standardized = new_df_standardized[
                            new_df_standardized["standard activity code"] != -1
                        ]
                        
                        generate_views(
                            new_df,
                            new_df_standardized,
                            dataset,
                            path_balanced=output_path / f"raw_balanced",
                            path_balanced_standardized=output_path / f"{view_name}",
                        )
                except Exception as e:
                    print(
                        f"Error generating the view {view_name} for {dataset}: {e}"
                    )
                    traceback.print_exc()
                    continue

    # Remove the junk folder
    workspace = Path("data/processed")
    if os.path.isdir(workspace):
        shutil.rmtree(workspace)
    # Remove the realworld2016_dataset_organized folder
    workspace = Path("data/original/RealWorld/realworld2016_dataset_organized")
    if os.path.isdir(workspace):
        shutil.rmtree(workspace)


if __name__ == "__main__":    
    choices = [
        "KuHar",
        "MotionSense",
        "WISDM",
        "UCI",
        "RealWorld",
    ]
    
    parser = argparse.ArgumentParser(description="Dataset Generator")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to process. If not provided, all datasets will be processed.",
        choices=choices,
        required=False,
        nargs="+",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the datasets",
        default="data/views",
        required=False,
    )
    
    args = parser.parse_args()
    datasets_to_process = choices
    if args.dataset:
        datasets_to_process = args.dataset

        
    print(f"Datasets to process: {datasets_to_process}")
    main(datasets_to_process, args.output_path)
