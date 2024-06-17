from pathlib import Path
import os, shutil
import pandas as pd
from typing import Tuple, List, Dict, Union

from dataset_processor import (
    AddGravityColumn,
    ButterworthFilter,
    Convert_G_to_Ms2,
    ResamplerPoly,
    Windowize,
    AddStandardActivityCode,
    SplitGuaranteeingAllClassesPerSplit,
    BalanceToMinimumClass,
    BalanceToMinimumClassAndUser,
    FilterByCommonRows,
    RenameColumns,
    Pipeline,
    Resampler,
    Interp1D
)

# Set the seed for reproducibility
import numpy as np
import random
import traceback

np.random.seed(42)
random.seed(42)
# pd.np.random.seed(42)

from utils import (
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
    1. ../data/unbalanced: The unbalanced dataset
    2. ../data/raw_balanced: The raw balanced dataset per activity
    3. ../data/standardized_balanced: The standardized balanced dataset per activity
    4. ../data/raw_balanced_user: The raw balanced dataset per user and activity
    5. ../data/standardized_balanced_user: The standardized balanced dataset per user and activity

    The datasets are generated in the following format:
    1. ../data/unbalanced/{dataset}/unbalanced.csv: The unbalanced dataset
    2. ../data/raw_balanced/{dataset}/train.csv: The raw balanced train dataset per activity
    3. ../data/raw_balanced/{dataset}/validation.csv: The raw balanced validation dataset per activity
    4. ../data/raw_balanced/{dataset}/test.csv: The raw balanced test dataset per activity
    5. ../data/standardized_balanced/{dataset}/train.csv: The standardized balanced train dataset per activity
    6. ../data/standardized_balanced/{dataset}/validation.csv: The standardized balanced validation dataset per activity
    7. ../data/standardized_balanced/{dataset}/test.csv: The standardized balanced test dataset per activity
    8. ../data/raw_balanced_user/{dataset}/train.csv: The raw balanced train dataset per user and activity
    9. ../data/raw_balanced_user/{dataset}/validation.csv: The raw balanced validation dataset per user and activity
    10. ../data/raw_balanced_user/{dataset}/test.csv: The raw balanced test dataset per user and activity
    11. ../data/standardized_balanced_user/{dataset}/train.csv: The standardized balanced train dataset per user and activity
    12. ../data/standardized_balanced_user/{dataset}/validation.csv: The standardized balanced validation dataset per user and activity
    13. ../data/standardized_balanced_user/{dataset}/test.csv: The standardized balanced test dataset per user and activity
"""
# Variables used to map the activities from the RealWorld dataset to the standard activities
maping: List[int] = [4, 3, -1, -1, 5, 0, 1, 2]
tasks: List[str] = [
    "climbingdown",
    "climbingup",
    "jumping",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]
standard_activity_code_realworld_map: Dict[str, int] = {
    activity: maping[tasks.index(activity)] for activity in tasks
}

datasets: List[str] = [
    # "KuHar",
    "MotionSense",
    # "WISDM",
    # "UCI",
    # "RealWorld",
]

column_group: Dict[str, str] = {
    "KuHar": "csv",
    "MotionSense": "csv",
    "WISDM": ["user", "activity code", "window"],
    "UCI": ["user", "activity code", "serial"],
    "RealWorld": ["user", "activity code", "position"],
}

standard_activity_code_map: Dict[str, Dict[Union[str, int], int]] = {
    "KuHar": {
        0: 1,
        1: 0,
        2: -1,
        3: -1,
        4: -1,
        5: -1,
        6: -1,
        7: -1,
        8: -1,
        9: -1,
        10: -1,
        11: 2,
        12: -1,
        13: -1,
        14: 5,
        15: 3,
        16: 4,
        17: -1,
    },
    "MotionSense": {0: 4, 1: 3, 2: 0, 3: 1, 4: 2, 5: 5},
    "WISDM": {
        "A": 2,
        "B": 5,
        "C": -1,
        "D": 0,
        "E": 1,
        "F": -1,
        "G": -1,
        "H": -1,
        "I": -1,
        "J": -1,
        "K": -1,
        "L": -1,
        "M": -1,
        "O": -1,
        "P": -1,
        "Q": -1,
        "R": -1,
        "S": -1,
    },
    "UCI": {
        1: 2,  # walk
        2: 3,  # stair up
        3: 4,  # stair down
        4: 0,  # sit
        5: 1,  # stand
        6: -1,  # Laying
        7: -1,  # stand to sit
        8: -1,  # sit to stand
        9: -1,  # sit to lie
        10: -1,  # lie to sit
        11: -1,  # stand to lie
        12: -1,  # lie to stand
    },
    "RealWorld": standard_activity_code_realworld_map,
}

standard_activity_code_names: Dict[int, str] = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}

columns_to_rename = {
    "KuHar": None,
    "MotionSense": {
        "userAcceleration.x": "accel-x",
        "userAcceleration.y": "accel-y",
        "userAcceleration.z": "accel-z",
        "rotationRate.x": "gyro-x",
        "rotationRate.y": "gyro-y",
        "rotationRate.z": "gyro-z",
    },
    "WISDM": None,
    "UCI": None,
    "RealWorld": None,
}

feature_columns: Dict[str, List[str]] = {
    "KuHar": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "MotionSense": [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
        "attitude.roll",
        "attitude.pitch",
        "attitude.yaw",
        "gravity.x",
        "gravity.y",
        "gravity.z",
    ],
    "WISDM": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "UCI": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "RealWorld": [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ],
}

match_columns: Dict[str, List[str]] = {
    "KuHar": ["user", "serial", "window", "activity code"],
    "MotionSense": ["user", "serial", "window"],
    "WISDM": ["user", "activity code", "window"],
    "UCI": ["user", "serial", "window", "activity code"],
    "RealWorld": ["user", "window", "activity code", "position"],
    "RealWorld_thigh": ["user", "window", "activity code", "position"],
    "RealWorld_upperarm": ["user", "window", "activity code", "position"],
    "RealWorld_waist": ["user", "window", "activity code", "position"],
}

# Pipelines to preprocess the datasets
pipelines: Dict[str, Dict[str, Pipeline]] = {
    # Kuhar Pipelines
    "KuHar": {
        # KuHar Raw
        "raw_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["KuHar"],
                    samples_per_window=300,
                    samples_per_overlap=0,
                    groupby_column=column_group["KuHar"],
                ),
                AddStandardActivityCode(standard_activity_code_map["KuHar"]),
            ]
        ),
        # KuHar standardized dataset 20hz with poly resample
        "standardized_dataset_20hz_poly_resample": Pipeline(
            [
                ResamplerPoly(
                    features_to_select=feature_columns["KuHar"],
                    up=1,
                    down=5,
                    groupby_column=column_group["KuHar"],
                ),
                Windowize(
                    features_to_select=feature_columns["KuHar"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["KuHar"],
                ),
                AddStandardActivityCode(standard_activity_code_map["KuHar"]),
            ]
        ),
    },

    # MotionSense Pipelines
    "MotionSense": {
        # MotionSense Raw
        "raw_dataset": Pipeline(
            [
                RenameColumns(columns_map=columns_to_rename["MotionSense"]),
                Windowize(
                    features_to_select=feature_columns["MotionSense"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["MotionSense"],
                ),
                AddStandardActivityCode(
                    standard_activity_code_map["MotionSense"]
                ),
            ]
        ),
        # MotionSense standardized dataset 20hz with poly resample
        "standardized_dataset_20hz_poly_resample": Pipeline(
            [
                RenameColumns(columns_map=columns_to_rename["MotionSense"]),
                AddGravityColumn(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    gravity_columns=["gravity.x", "gravity.y", "gravity.z"],
                ),
                Convert_G_to_Ms2(
                    axis_columns=["accel-x", "accel-y", "accel-z"]
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["MotionSense"],
                    up=2,
                    down=5,
                    groupby_column=column_group["MotionSense"],
                ),
                Windowize(
                    features_to_select=feature_columns["MotionSense"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["MotionSense"],
                ),
                AddStandardActivityCode(
                    standard_activity_code_map["MotionSense"]
                ),
            ]
        ),
 
    },
    
    # WISDM Pipelines
    "WISDM": {
        # WISDM Raw
        "raw_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["WISDM"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["WISDM"],
                ),
                AddStandardActivityCode(standard_activity_code_map["WISDM"]),
            ]
        ),
        # WISDM standardized dataset 20hz with poly resample
        "standardized_dataset_20hz_poly_resample": Pipeline(
            [
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=20,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["WISDM"],
                    up=1,
                    down=1,
                    groupby_column=column_group["WISDM"],
                ),
                Windowize(
                    features_to_select=feature_columns["WISDM"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["WISDM"],
                ),
                AddStandardActivityCode(standard_activity_code_map["WISDM"]),
            ]
        ),
    },
    
    # UCI Pipelines
    "UCI": {
        # UCI raw
        "raw_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["UCI"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["UCI"],
                ),
                AddStandardActivityCode(standard_activity_code_map["UCI"]),
            ]
        ),
        # UCI standardized dataset 20hz with poly resample
        "standardized_dataset_20hz_poly_resample": Pipeline(
            [
                Convert_G_to_Ms2(
                    axis_columns=["accel-x", "accel-y", "accel-z"]
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["UCI"],
                    up=2,
                    down=5,
                    groupby_column=column_group["UCI"],
                ),
                Windowize(
                    features_to_select=feature_columns["UCI"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["UCI"],
                ),
                AddStandardActivityCode(standard_activity_code_map["UCI"]),
            ]
        ),
    },
    
    # RealWorld Pipelines
    "RealWorld": {
        # + RealWorld Raw
        "raw_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["RealWorld"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["RealWorld"],
                ),
                AddStandardActivityCode(
                    standard_activity_code_map["RealWorld"]
                ),
            ]
        ),
        # + RealWorld standardized dataset 20hz with poly resample
        "standardized_dataset_20hz_poly_resample": Pipeline(
            [
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["RealWorld"],
                    up=2,
                    down=5,
                    groupby_column=column_group["RealWorld"],
                ),
                Windowize(
                    features_to_select=feature_columns["RealWorld"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["RealWorld"],
                ),
                AddStandardActivityCode(
                    standard_activity_code_map["RealWorld"]
                ),
            ]
        ),

    },
}

# Creating a list of functions to read the datasets
functions: Dict[str, callable] = {
    "KuHar": read_kuhar,
    "MotionSense": read_motionsense,
    "WISDM": read_wisdm,
    "UCI": read_uci,
    "RealWorld": read_realworld,
}

dataset_path: Dict[str, str] = {
    "KuHar": "KuHar/1.Raw_time_domian_data",
    "MotionSense": "MotionSense/A_DeviceMotion_data",
    "WISDM": "WISDM/wisdm-dataset/raw/phone",
    "UCI": "UCI/RawData",
    "RealWorld": "RealWorld/realworld2016_dataset",
}

# Preprocess the datasets

# Path to save the datasets
output_path: Path = Path("../data/datasets")
# output_path_unbalanced: object = Path("../data/unbalanced")

# output_path_balanced: object = Path("../data/raw_balanced")
# output_path_balanced_standardized: object = Path("../data/standardized_balanced")

# output_path_balanced_user: object = Path("../data/raw_balanced_user")
# output_path_balanced_standardized_user: object = Path("../data/standardized_balanced_user")

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
    path_unbalanced,
    path_balanced_user,
    path_balanced,
    path_balanced_standardized_user,
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

    # Save the unbalanced dataset
    output_dir = path_unbalanced / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_dir / "unbalanced.csv", index=False)

    # Preprocess and save the raw balanced dataset per user and activity
    train_df, val_df, test_df = balance_per_user_and_activity(
        dataset, new_df, path_balanced_user
    )
    sanity_function(train_df, val_df, test_df)

    # Preprocess and save the raw balanced dataset per activity
    train_df, val_df, test_df = balance_per_activity(
        dataset, new_df, path_balanced
    )
    sanity_function(train_df, val_df, test_df)

    # Preprocess and save the standardized balanced dataset per user and activity
    train_df, val_df, test_df = balance_per_user_and_activity(
        dataset, new_df_standardized, path_balanced_standardized_user
    )
    sanity_function(train_df, val_df, test_df)

    # Preprocess and save the standardized balanced dataset per activity
    train_df, val_df, test_df = balance_per_activity(
        dataset, new_df_standardized, path_balanced_standardized
    )
    sanity_function(train_df, val_df, test_df)


def main():
    # Creating the datasets
    for dataset in datasets:
        print(f"Preprocess the dataset {dataset} ...\n")

        reader = functions[dataset]

        # Read the raw dataset
        if dataset == "RealWorld":
            print("Organizing the RealWorld dataset ...\n")
            # Create a folder to save the organized dataset
            workspace = Path(
                "../data/original/RealWorld/realworld2016_dataset_organized"
            )
            if not os.path.isdir(workspace):
                os.mkdir(workspace)
            # Organize the dataset
            workspace, users = real_world_organize()
            path = workspace
            raw_dataset = reader(path, users)
            # Preprocess the raw dataset
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
                            path_unbalanced=output_path / f"raw_unbalanced",
                            path_balanced_user=output_path / f"raw_balanced_user",
                            path_balanced=output_path / f"raw_balanced",
                            path_balanced_standardized_user=output_path / f"raw_balanced_user_standardized",
                            path_balanced_standardized=output_path / f"raw_balanced_standardized",
                        )
                        positions = new_df["position"].unique()
                        for position in list(positions):
                            new_df_filtered = new_df[new_df["position"] == position]
                            new_df_standardized_filtered = new_df_standardized[
                                new_df_standardized["position"] == position
                            ]
                            new_dataset = dataset + "_" + position
                            generate_views(
                                new_df_filtered,
                                new_df_standardized_filtered,
                                new_dataset,
                                path_unbalanced=output_path
                                / f"{view_name}_unbalanced",
                                path_balanced_user=output_path
                                / f"{view_name}_balanced_user",
                                path_balanced=output_path / f"{view_name}_balanced",
                                path_balanced_standardized_user=output_path
                                / f"{view_name}_balanced_user_standardized",
                                path_balanced_standardized=output_path
                                / f"{view_name}_balanced_standardized",
                            )
                except Exception as e:
                    print(f"Error generating the view {view_name} for {dataset}: {e}")
                    traceback.print_exc()
                    continue
        else:
            path = Path(f"../data/original/{dataset_path[dataset]}")
            raw_dataset = reader(path)
            # Preprocess the raw dataset
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
                            path_unbalanced=output_path / f"{view_name}_unbalanced",
                            path_balanced_user=output_path
                            / f"{view_name}_balanced_user",
                            path_balanced=output_path / f"{view_name}_balanced",
                            path_balanced_standardized_user=output_path
                            / f"{view_name}_balanced_user_standardized",
                            path_balanced_standardized=output_path
                            / f"{view_name}_balanced_standardized",
                        )
                except Exception as e:
                    print(f"Error generating the view {view_name} for {dataset}: {e}")
                    traceback.print_exc()
                    continue

    # Remove the junk folder
    workspace = Path("../data/processed")
    if os.path.isdir(workspace):
        shutil.rmtree(workspace)
    # Remove the realworld2016_dataset_organized folder
    workspace = Path(
        "../data/original/RealWorld/realworld2016_dataset_organized"
    )
    if os.path.isdir(workspace):
        shutil.rmtree(workspace)


if __name__ == "__main__":
    main()
