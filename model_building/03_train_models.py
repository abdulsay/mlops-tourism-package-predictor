"""
Activity 3: Model Training and Experimentation

Models:
1. Decision Tree
2. Bagging
3. Random Forest
4. XGBoost

Purpose:
- Load training data
- Build preprocessing pipeline
- Tune hyperparameters
- Compare models
- Select best model using CV F1 score
- Save all tuned models
- Save best model
"""

from pathlib import Path
import json
import pickle

import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import (
    ColumnTransformer
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.impute import (
    SimpleImputer
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier
)

import xgboost as xgb

from sklearn.model_selection import (
    GridSearchCV
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TRAIN_FILE = Path(
    "tourism_project/artifacts/train.csv"
)

MODEL_DIRECTORY = Path(
    "tourism_project/models/"
)

RESULTS_FILE = Path(
    "tourism_project/artifacts/model_results.csv"
)

TARGET_COLUMN = "ProdTaken"

RANDOM_STATE = 42


# ---------------------------------------------------------
# Create preprocessing pipeline
# ---------------------------------------------------------

def create_preprocessor(
    numerical_columns,
    categorical_columns
):

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# Define models and parameters
# ---------------------------------------------------------

def get_models():

    models = {

        "DecisionTree": {

            "model": DecisionTreeClassifier(
                random_state=RANDOM_STATE
            ),

            "parameters": {

                "model__max_depth":
                    [4, 6, 10, None],

                "model__min_samples_split":
                    [2, 5, 10],

                "model__min_samples_leaf":
                    [1, 2, 5]
            }
        },

        "Bagging": {

            "model": BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    random_state=RANDOM_STATE
                ),
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

            "parameters": {

                "model__n_estimators":
                    [50, 100, 200],

                "model__max_samples":
                    [0.7, 1.0],

                "model__max_features":
                    [0.7, 1.0]
            }
        },

        "RandomForest": {

            "model": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

            "parameters": {

                "model__n_estimators":
                    [100, 200],

                "model__max_depth":
                    [8, 12, None],

                "model__min_samples_split":
                    [2, 5],

                "model__min_samples_leaf":
                    [1, 2]
            }
        },

        "XGBoost": {

            "model": xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss"
            ),

            "parameters": {

                "model__n_estimators":
                    [100, 200],

                "model__max_depth":
                    [3, 5, 7],

                "model__learning_rate":
                    [0.05, 0.1],

                "model__subsample":
                    [0.8, 1.0]
            }
        }
    }

    return models


# ---------------------------------------------------------
# Train models
# ---------------------------------------------------------

def train_models():

    print("=" * 60)
    print("MODEL TRAINING AND EXPERIMENTATION")
    print("=" * 60)

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Load training data
    # -----------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = train_df[TARGET_COLUMN]

    # -----------------------------------------------------
    # Identify feature types
    # -----------------------------------------------------

    numerical_columns = (
        X_train
        .select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X_train
        .select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )

    print(
        f"\nNumerical features: "
        f"{len(numerical_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    # -----------------------------------------------------
    # Create preprocessing
    # -----------------------------------------------------

    preprocessor = create_preprocessor(
        numerical_columns,
        categorical_columns
    )

    # -----------------------------------------------------
    # Get models
    # -----------------------------------------------------

    models = get_models()

    results = []

    best_model = None
    best_model_name = None
    best_cv_score = -1
    best_parameters = None

    # -----------------------------------------------------
    # Experiment with each model
    # -----------------------------------------------------

    for model_name, configuration in models.items():

        print("\n" + "=" * 60)

        print(
            f"Training: {model_name}"
        )

        print("=" * 60)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    configuration["model"]
                )
            ]
        )

        grid_search = GridSearchCV(

            estimator=pipeline,

            param_grid=configuration["parameters"],

            scoring="f1",

            cv=3,

            n_jobs=-1,

            refit=True
        )

        # Train
        grid_search.fit(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # Best CV score
        # -------------------------------------------------

        cv_score = (
            grid_search.best_score_
        )

        print(
            f"\nBest CV F1: "
            f"{cv_score:.4f}"
        )

        print(
            "\nBest parameters:"
        )

        print(
            grid_search.best_params_
        )

        # -------------------------------------------------
        # Save tuned model
        # -------------------------------------------------

        model_file = (
            MODEL_DIRECTORY /
            f"{model_name.lower()}.pkl"
        )

        with open(
            model_file,
            "wb"
        ) as file:

            pickle.dump(
                grid_search.best_estimator_,
                file
            )

        # -------------------------------------------------
        # Store results
        # -------------------------------------------------

        results.append({

            "model":
                model_name,

            "cv_f1_score":
                cv_score,

            "best_parameters":
                json.dumps(
                    grid_search.best_params_,
                    default=str
                )
        })

        # -------------------------------------------------
        # Select best model
        #
        # IMPORTANT:
        # Test data is NOT used here.
        # -------------------------------------------------

        if cv_score > best_cv_score:

            best_cv_score = cv_score

            best_model = (
                grid_search.best_estimator_
            )

            best_model_name = (
                model_name
            )

            best_parameters = (
                grid_search.best_params_
            )

    # -----------------------------------------------------
    # Save best model
    # -----------------------------------------------------

    best_model_file = (
        MODEL_DIRECTORY /
        "best_model.pkl"
    )

    with open(
        best_model_file,
        "wb"
    ) as file:

        pickle.dump(
            best_model,
            file
        )

    # -----------------------------------------------------
    # Save experiment results
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "cv_f1_score",
        ascending=False
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Save metadata
    # -----------------------------------------------------

    metadata = {

        "best_model":
            best_model_name,

        "selection_metric":
            "Cross-validation F1",

        "best_cv_f1_score":
            best_cv_score,

        "best_parameters":
            best_parameters,

        "numerical_columns":
            numerical_columns,

        "categorical_columns":
            categorical_columns
    }

    metadata_file = (
        MODEL_DIRECTORY /
        "model_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            default=str
        )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL EXPERIMENTATION COMPLETED")
    print("=" * 60)

    print(
        f"\nSelected model: "
        f"{best_model_name}"
    )

    print(
        f"Best CV F1: "
        f"{best_cv_score:.4f}"
    )

    print("\nModel comparison:")

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    train_models()