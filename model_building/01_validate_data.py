"""
Activity 1: Dataset Validation

Purpose:
- Load the dataset from the repository
- Validate expected columns
- Check missing values
- Check duplicate records
- Display target distribution
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = Path("tourism_project/data/tourism.csv")

EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_data():

    print("=" * 60)
    print("TOURISM DATASET VALIDATION")
    print("=" * 60)

    # Check file
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset: {DATA_PATH}")
    print(f"Rows   : {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # -----------------------------------------------------
    # Validate columns
    # -----------------------------------------------------

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    print("\nColumn validation: PASSED")

    # -----------------------------------------------------
    # Extra columns
    # -----------------------------------------------------

    extra_columns = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    if extra_columns:
        print(
            f"Additional columns: {extra_columns}"
        )

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    total_missing = df.isnull().sum().sum()

    print("\nMissing values:", total_missing)

    if total_missing > 0:

        print(
            df.isnull()
            .sum()
            .loc[lambda x: x > 0]
        )

    # -----------------------------------------------------
    # Duplicate records
    # -----------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print(
        f"\nDuplicate records: {duplicate_count}"
    )

    # -----------------------------------------------------
    # Target distribution
    # -----------------------------------------------------

    print("\nTarget distribution:")
    print(
        df["ProdTaken"]
        .value_counts()
        .sort_index()
    )

    print("\nTarget percentage:")
    print(
        (
            df["ProdTaken"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\nDataset summary:")
    print(
        df.describe(include="all").transpose()
    )

    print("\n" + "=" * 60)
    print("DATA VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    validate_data()