"""
Activity 4: Model Evaluation

Purpose:
- Load unseen test dataset
- Load each tuned model
- Generate predictions
- Display classification_report
- Compare final test performance

The test set is NOT used to select hyperparameters.
"""

from pathlib import Path
import pickle

import pandas as pd

from sklearn.metrics import classification_report


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TEST_FILE = Path(
    "tourism_project/artifacts/test.csv"
)

MODEL_DIRECTORY = Path(
    "tourism_project/models/"
)

TARGET_COLUMN = "ProdTaken"


MODEL_FILES = {

    "DecisionTree":
        "decisiontree.pkl",

    "Bagging":
        "bagging.pkl",

    "RandomForest":
        "randomforest.pkl",

    "XGBoost":
        "xgboost.pkl"
}


# ---------------------------------------------------------
# Evaluate models
# ---------------------------------------------------------

def evaluate_models():

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------

    test_df = pd.read_csv(
        TEST_FILE
    )

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_test = test_df[TARGET_COLUMN]

    # -----------------------------------------------------
    # Evaluate each model
    # -----------------------------------------------------

    for model_name, file_name in MODEL_FILES.items():

        model_file = (
            MODEL_DIRECTORY /
            file_name
        )

        print("\n" + "=" * 60)

        print(
            f"MODEL: {model_name}"
        )

        print("=" * 60)

        # Load model
        with open(
            model_file,
            "rb"
        ) as file:

            model = pickle.load(
                file
            )

        # Generate predictions
        predictions = model.predict(
            X_test
        )

        # Classification report
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )


if __name__ == "__main__":
    evaluate_models()