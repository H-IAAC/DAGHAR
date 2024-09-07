from typing import List, Dict, Union

from steps import (
    AddGravityColumn,
    ButterworthFilter,
    Convert_G_to_Ms2,
    ResamplerPoly,
    Windowize,
    AddStandardActivityCode,
    RenameColumns,
    Pipeline,
)

# Set the seed for reproducibility

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
        "standardized_dataset": Pipeline(
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
        "standardized_dataset": Pipeline(
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
        "standardized_dataset": Pipeline(
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
        "standardized_dataset": Pipeline(
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
        "standardized_dataset": Pipeline(
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
