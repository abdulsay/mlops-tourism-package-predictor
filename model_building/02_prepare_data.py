"""
Activity 2: Data Preparation

Purpose:
- Load source dataset
- Remove unnecessary columns
- Clean data
- Split into training and testing datasets
- Save train.csv and test.csv
"""

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SOURCE_FILE = Path("data/tourism.csv")

OUTPUT_DIRECTORY = Path("artifacts")

TRAIN_FILE = OUTPUT_DIRECTORY / "train.csv"

TEST_FILE = OUTPUT_DIRECTORY / "test.csv"

TARGET_COLUMN = "ProdTaken"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ---------------------------------------------------------
# Data preparation
# ---------------------------------------------------------

def prepare_data():

    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    df = pd.read_csv(
        SOURCE_FILE
    )

    print(
        f"\nOriginal dataset shape: {df.shape}"
    )

    # -----------------------------------------------------
    # Remove unnecessary columns
    # -----------------------------------------------------

    columns_to_remove = [
        "Unnamed: 0",
        "CustomerID"
    ]

    df = df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    print(
        f"After removing unnecessary columns: {df.shape}"
    )

    # -----------------------------------------------------
    # Clean categorical values
    # -----------------------------------------------------

    if "Gender" in df.columns:

        df["Gender"] = (
            df["Gender"]
            .replace(
                {
                    "Fe Male": "Female"
                }
            )
        )

    # -----------------------------------------------------
    # Remove duplicate rows
    # -----------------------------------------------------

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:

        print(
            f"Removing {duplicate_count} duplicates"
        )

        df = df.drop_duplicates()

    # -----------------------------------------------------
    # Separate features and target
    # -----------------------------------------------------

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )

    # -----------------------------------------------------
    # Recreate train/test DataFrames
    # -----------------------------------------------------

    train_df = X_train.copy()

    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()

    test_df[TARGET_COLUMN] = y_test

    # -----------------------------------------------------
    # Save datasets
    # -----------------------------------------------------

    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\nData preparation completed.")

    print(
        f"Training records: {len(train_df)}"
    )

    print(
        f"Testing records : {len(test_df)}"
    )

    print(
        f"\nTraining file: {TRAIN_FILE}"
    )

    print(
        f"Testing file : {TEST_FILE}"
    )


if __name__ == "__main__":
    prepare_data()
