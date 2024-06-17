from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

from dataset_processor import BalanceToMinimumClass, BalanceToMinimumClassAndUser

"""This module generates the balanced data by activity and user, and saves them in the `output_dir` folder.
The difference between this module is that it balances the data by minimum class count and minimum number of users for dataset.
Therefore, the number of samples in each activity and user is the same in each split for each view for each dataset.

The data is read from the `root_dir` folder, and the balanced data is saved in the `output_dir` folder."""

# Location with the data
root_dir = Path("../data/standartized_balanced")
root_dir_user = Path("../data/standartized_balanced_user")
# Location to save the data
output_dir = Path("../data/standartized_inter_balanced")
output_dir_user = Path("../data/standartized_inter_balanced_user")

# Class to balance
class_to_balance: str = "standard activity code"

standartized_codes: Dict[int, str] = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}

# Get the class codes
class_codes: List[int] = list(standartized_codes.keys())

def balance_by_activity(root_dir, output_dir):
    """This function balances the data by activity, so that each activity has the same number of samples in each split.

    Parameters
    ----------
    root_dir : Path
        Path to the folder with the data.
    output_dir : Path
        Path to the folder where the balanced data will be saved.

    Returns
    -------
    None
        The balanced data is saved in the `output_dir` folder.
    """

    # Minimum of each split 
    split_min = {"train": np.inf, "validation": np.inf, "test": np.inf}

    # Read all CSVs from all datasets
    for f in root_dir.rglob("*.csv"):
        # Read dataframe
        df = pd.read_csv(f)
        # For each class `c`
        for c in class_codes:
            # Get the split name, based on file name (train, validation or test)
            split_name = f.stem
            # Number of elements from class `c`
            numel = len(df[df[class_to_balance] == c])
            # If the dataset does not have any element from class `c`, skip it
            if numel > 0:
                # Update the minimum
                split_min[split_name] = min(split_min[split_name], numel)

    # Create a dictionary with the minimum class count for each split
    split_balancer = {
        "train": BalanceToMinimumClass(
            class_column=class_to_balance, min_value=split_min["train"], random_state=0
        ),
        "validation": BalanceToMinimumClass(
            class_column=class_to_balance, min_value=split_min["validation"], random_state=0
        ),
        "test": BalanceToMinimumClass(
            class_column=class_to_balance, min_value=split_min["test"], random_state=0
        ),
    }

    # Dump some information
    print("Minimum class count in each split (from all files):")
    print(split_min)

    split(root_dir, split_balancer, output_dir)


def balance_by_user(root_dir, output_dir):
    """This function balances the data by activity and user, so that each activity has the same number of samples in each split.
    
    Parameters
    ----------
    root_dir : Path
        Path to the folder with the data.
    output_dir : Path
        Path to the folder where the balanced data will be saved.

    Returns
    -------
    None
        The balanced data is saved in the `output_dir` folder.
    """

    # Minimum of each split 
    split_min = {"train": np.inf, "validation": np.inf, "test": np.inf}

    min_users = {"train": np.inf, "validation": np.inf, "test": np.inf}

    # Read all CSVs from all datasets
    for f in root_dir.rglob("*.csv"):
        # Read dataframe
        df = pd.read_csv(f)
        users = df["user"].unique()
        # For each class `c`
        for c in class_codes:
            # Get the split name, based on file name (train, validation or test)
            split_name = f.stem
            # Number of elements from class `c` and user `u`
            for u in users:
                numel = len(df[(df[class_to_balance] == c) & (df["user"] == u)])
                # If the dataset does not have any element from class `c`, skip it
                if numel > 0:
                    # Update the minimum
                    split_min[split_name] = min(split_min[split_name], numel)
                    min_users[split_name] = min(min_users[split_name], len(users))

    # Create a dictionary with the minimum class count for each split
    split_balancer = {
        "train": BalanceToMinimumClassAndUser(
            class_column=class_to_balance, min_value=split_min["train"], random_state=0
        ),
        "validation": BalanceToMinimumClassAndUser(
            class_column=class_to_balance, min_value=split_min["validation"], random_state=0
        ),
        "test": BalanceToMinimumClassAndUser(
            class_column=class_to_balance, min_value=split_min["test"], random_state=0
        ),
    }

    # Dump some information
    print("Minimum class count in each split (from all files):")
    print(split_min)

    print("Minimum number of users in each split (from all files):")
    print(min_users)

    split(root_dir, split_balancer, output_dir, min_users)

def split(root_dir, split_balancer, output_dir, min_users: dict = None):
    """This function splits the data into train, validation and test sets, and saves them in the `output_dir` folder.

    Parameters
    ----------
    root_dir : Path
        Path to the folder with the data.
    split_balancer : dict
        Dictionary with the minimum class count for each split.
    output_dir : Path
        Path to the folder where the balanced data will be saved.
    min_users : dict, optional
        The minimum number of users in each split, by default None

    Returns
    -------
    None
        The balanced data is saved in the `output_dir` folder.
    """

    # Read all CSVs from all datasets
    for f in root_dir.rglob("*.csv"):
        # Get the dataset name, based on the parent folder name
        dataset_name = f.parent.name
        # Get the split name, based on file name (train, validation or test)
        split_name = f.stem
        # # Get the filename (without parent directories)
        # fname = f.name
        # Read dataframe
        df = pd.read_csv(f)
        # Select the minimun users in the split with seed = 0
        if min_users:
            users = df["user"].unique()
            np.random.seed(0)
            np.random.shuffle(users)
            users = users[:min_users[split_name]]
            df = df[df["user"].isin(users)]

        # Balance the dataframe (based on the minimum class count of that split)
        df = split_balancer[split_name](df)
        # Create the output filename
        output_fname =  output_dir / dataset_name / f"{split_name}.csv"
        # Create the output directory (if it does not exist)
        output_fname.parent.mkdir(exist_ok=True, parents=True)
        # Save the dataframe
        df.to_csv(output_fname, index=False)

print("Generating the balanced data by activity")
balance_by_activity(root_dir, output_dir)
print("\nGenerating the balanced data by activity and user")
balance_by_user(root_dir_user, output_dir_user)