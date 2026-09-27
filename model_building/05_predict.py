"""
Activity 5: Prediction

Purpose:
- Load the best trained model
- Collect customer information
- Create a DataFrame
- Generate a prediction
- Display purchase probability
"""

from pathlib import Path
import pickle

import pandas as pd


# =========================================================
# Configuration
# =========================================================

MODEL_FILE = Path(
    "tourism_project/models/best_model.pkl"
)


# =========================================================
# Load model
# =========================================================

def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Best model not found. "
            "Please run 03_train_models.py first."
        )

    with open(
        MODEL_FILE,
        "rb"
    ) as file:

        model = pickle.load(file)

    return model


# =========================================================
# Get customer input
# =========================================================

def get_customer_input():

    print("\n" + "=" * 60)
    print("CUSTOMER INFORMATION")
    print("=" * 60)

    customer = {

        "Age": int(
            input("Age: ")
        ),

        "TypeofContact": input(
            "Type of Contact "
            "(Self Enquiry/Company Invited): "
        ),

        "CityTier": int(
            input("City Tier (1/2/3): ")
        ),

        "DurationOfPitch": int(
            input("Duration of Pitch: ")
        ),

        "Occupation": input(
            "Occupation: "
        ),

        "Gender": input(
            "Gender (Male/Female): "
        ),

        "NumberOfPersonVisiting": int(
            input(
                "Number of Persons Visiting: "
            )
        ),

        "NumberOfFollowups": int(
            input(
                "Number of Follow-ups: "
            )
        ),

        "ProductPitched": input(
            "Product Pitched: "
        ),

        "PreferredPropertyStar": int(
            input(
                "Preferred Property Star (3/4/5): "
            )
        ),

        "MaritalStatus": input(
            "Marital Status: "
        ),

        "NumberOfTrips": int(
            input(
                "Number of Trips: "
            )
        ),

        "Passport": int(
            input(
                "Passport (0=No, 1=Yes): "
            )
        ),

        "PitchSatisfactionScore": int(
            input(
                "Pitch Satisfaction Score (1-5): "
            )
        ),

        "OwnCar": int(
            input(
                "Own Car (0=No, 1=Yes): "
            )
        ),

        "NumberOfChildrenVisiting": int(
            input(
                "Number of Children Visiting: "
            )
        ),

        "Designation": input(
            "Designation: "
        ),

        "MonthlyIncome": float(
            input(
                "Monthly Income: "
            )
        )
    }

    return customer


# =========================================================
# Prediction
# =========================================================

def predict():

    print("=" * 60)
    print("TOURISM PACKAGE PREDICTION")
    print("=" * 60)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = load_model()

    # -----------------------------------------------------
    # Get customer information
    # -----------------------------------------------------

    customer_data = get_customer_input()

    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    customer_df = pd.DataFrame(
        [customer_data]
    )

    print("\n" + "=" * 60)
    print("CUSTOMER DATA")
    print("=" * 60)

    print(
        customer_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Generate prediction
    # -----------------------------------------------------

    prediction = model.predict(
        customer_df
    )[0]

    # -----------------------------------------------------
    # Generate probability
    # -----------------------------------------------------

    probability = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probability = (
            model
            .predict_proba(
                customer_df
            )[0][1]
        )

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    if prediction == 1:

        print(
            "Prediction: CUSTOMER WILL "
            "LIKELY PURCHASE"
        )

    else:

        print(
            "Prediction: CUSTOMER IS "
            "UNLIKELY TO PURCHASE"
        )

    if probability is not None:

        print(
            f"Purchase Probability: "
            f"{probability:.2%}"
        )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    predict()