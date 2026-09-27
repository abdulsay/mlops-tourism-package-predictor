"""
Validate the Tourism Package Prediction dataset.

Checks:
- Dataset exists
- Expected columns are present
- Missing values
- Duplicate rows
- Target distribution
- Basic dataset summary
"""

from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/tourism.csv")

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
    "MonthlyIncome",
]


def validate_dataset():

    # Check if dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("TOURISM DATASET VALIDATION")
    print("=" * 60)

    print(f"Dataset path       : {DATA_PATH}")
    print(f"Number of rows     : {df.shape[0]}")
    print(f"Number of columns  : {df.shape[1]}")

    # ---------------------------------------------------------
    # Validate columns
    # ---------------------------------------------------------

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_columns = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    print("\nColumn validation : PASSED")

    if extra_columns:
        print(f"Additional columns: {extra_columns}")

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing_values = df.isnull().sum()

    print("\nMissing Values")
    print("-" * 40)

    if missing_values.sum() == 0:
        print("No missing values found.")
    else:
        print(
            missing_values[
                missing_values > 0
            ]
        )

    # ---------------------------------------------------------
    # Duplicate records
    # ---------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print("\nDuplicate Records")
    print("-" * 40)
    print(f"Duplicate rows: {duplicate_count}")

    # ---------------------------------------------------------
    # Target distribution
    # ---------------------------------------------------------

    print("\nTarget Distribution - ProdTaken")
    print("-" * 40)

    print(
        df["ProdTaken"]
        .value_counts()
        .sort_index()
    )

    print("\nTarget Percentage")
    print("-" * 40)

    print(
        df["ProdTaken"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    # ---------------------------------------------------------
    # Dataset information
    # ---------------------------------------------------------

    print("\nDataset Information")
    print("-" * 40)

    print(df.info())

    # ---------------------------------------------------------
    # Statistical summary
    # ---------------------------------------------------------

    print("\nStatistical Summary")
    print("-" * 40)

    print(
        df.describe(
            include="all"
        ).transpose()
    )

    print("\n" + "=" * 60)
    print("DATA VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    validate_dataset()