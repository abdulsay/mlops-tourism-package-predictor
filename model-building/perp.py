"""
Prepare Tourism Package Prediction data.

Steps:
1. Load dataset from repository data folder
2. Remove unnecessary columns
3. Clean categorical values
4. Split data into training and testing datasets
5. Save train/test datasets locally
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SOURCE_PATH = Path("tourism_project/data/tourism.csv")
OUTPUT_PATH = Path("artifacts")
TARGET_COLUMN = "ProdTaken"
TEST_SIZE = 0.30
RANDOM_STATE = 42


# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------

def prepare_data():

    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    # Create output directory
    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        SOURCE_PATH
    )

    print(
        f"Original dataset shape: {df.shape}"
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
    # Data cleaning
    # -----------------------------------------------------

    # Normalize gender values
    if "Gender" in df.columns:

        df["Gender"] = df[
            "Gender"
        ].replace(
            {
                "Fe Male": "Female"
            }
        )

    # -----------------------------------------------------
    # Remove duplicate records
    # -----------------------------------------------------

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:

        print(
            f"Removing {duplicate_count} duplicate records"
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
    # Train-test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )

    # -----------------------------------------------------
    # Recreate train/test datasets
    # -----------------------------------------------------

    train_df = X_train.copy()

    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()

    test_df[TARGET_COLUMN] = y_test

    # -----------------------------------------------------
    # Save files
    # -----------------------------------------------------

    train_path = OUTPUT_PATH / "train.csv"

    test_path = OUTPUT_PATH / "test.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\nData Preparation Completed")

    print(
        f"Training records: {len(train_df)}"
    )

    print(
        f"Testing records : {len(test_df)}"
    )

    print(
        f"Train target rate: "
        f"{train_df[TARGET_COLUMN].mean():.4f}"
    )

    print(
        f"Test target rate : "
        f"{test_df[TARGET_COLUMN].mean():.4f}"
    )

    print(
        f"\nTraining file: {train_path}"
    )

    print(
        f"Testing file : {test_path}"
    )


if __name__ == "__main__":

    prepare_data()