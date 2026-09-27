"""
Tourism Package Prediction
Streamlit Deployment Application

The application:
1. Loads the trained model from the repository
2. Collects customer information
3. Creates a Pandas DataFrame
4. Generates a purchase prediction
5. Displays the prediction and probability
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = Path(
    "models/best_model.pkl"
)


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)


# =========================================================
# Load trained model
# =========================================================

@st.cache_resource
def load_model():

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        model = pickle.load(file)

    return model


model = load_model()


# =========================================================
# Application title
# =========================================================

st.title(
    "✈️ Tourism Package Purchase Prediction"
)

st.write(
    """
    **Visit with Us**

    This application predicts whether a customer is likely
    to purchase the Wellness Tourism Package.
    """
)


st.divider()


# =========================================================
# Customer Information
# =========================================================

st.subheader(
    "Customer Information"
)


col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# Column 1
# ---------------------------------------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    city_tier = st.selectbox(
        "City Tier",
        options=[1, 2, 3],
        index=1
    )

    occupation = st.selectbox(
        "Occupation",
        options=[
            "Salaried",
            "Free Lancer",
            "Small Business",
            "Large Business"
        ]
    )

    gender = st.selectbox(
        "Gender",
        options=[
            "Male",
            "Female"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        options=[
            "Married",
            "Single",
            "Divorced"
        ]
    )


# ---------------------------------------------------------
# Column 2
# ---------------------------------------------------------

with col2:

    type_of_contact = st.selectbox(
        "Type of Contact",
        options=[
            "Self Enquiry",
            "Company Invited"
        ]
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=0,
        max_value=120,
        value=15,
        step=1
    )

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    number_of_followups = st.number_input(
        "Number of Follow-ups",
        min_value=0,
        max_value=10,
        value=3,
        step=1
    )

    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=20,
        value=3,
        step=1
    )


# ---------------------------------------------------------
# Column 3
# ---------------------------------------------------------

with col3:

    product_pitched = st.selectbox(
        "Product Pitched",
        options=[
            "Basic",
            "Standard",
            "Deluxe",
            "Super Deluxe",
            "King"
        ]
    )

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        options=[
            3,
            4,
            5
        ]
    )

    passport = st.selectbox(
        "Passport",
        options=[
            0,
            1
        ],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    own_car = st.selectbox(
        "Own Car",
        options=[
            0,
            1
        ],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )


# =========================================================
# Additional Information
# =========================================================

st.subheader(
    "Additional Information"
)


col4, col5, col6 = st.columns(3)


with col4:

    pitch_satisfaction_score = st.slider(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )


with col5:

    designation = st.selectbox(
        "Designation",
        options=[
            "AVP",
            "Executive",
            "Manager",
            "Senior Manager",
            "VP"
        ]
    )


with col6:

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        max_value=1000000,
        value=25000,
        step=1000
    )


# =========================================================
# Prediction
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Purchase",
    type="primary"
)


if predict_button:

    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    customer_data = {

        "Age": age,

        "TypeofContact":
            type_of_contact,

        "CityTier":
            city_tier,

        "DurationOfPitch":
            duration_of_pitch,

        "Occupation":
            occupation,

        "Gender":
            gender,

        "NumberOfPersonVisiting":
            number_of_person_visiting,

        "NumberOfFollowups":
            number_of_followups,

        "ProductPitched":
            product_pitched,

        "PreferredPropertyStar":
            preferred_property_star,

        "MaritalStatus":
            marital_status,

        "NumberOfTrips":
            number_of_trips,

        "Passport":
            passport,

        "PitchSatisfactionScore":
            pitch_satisfaction_score,

        "OwnCar":
            own_car,

        "NumberOfChildrenVisiting":
            number_of_children_visiting,

        "Designation":
            designation,

        "MonthlyIncome":
            monthly_income
    }

    customer_df = pd.DataFrame(
        [customer_data]
    )

    # -----------------------------------------------------
    # Display input DataFrame
    # -----------------------------------------------------

    st.subheader(
        "Customer Data"
    )

    st.dataframe(
        customer_df,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Generate prediction
    # -----------------------------------------------------

    prediction = model.predict(
        customer_df
    )[0]

    # -----------------------------------------------------
    # Purchase probability
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

    st.subheader(
        "Prediction Result"
    )

    if prediction == 1:

        st.success(
            "🎉 The customer is likely to purchase the "
            "Wellness Tourism Package."
        )

    else:

        st.info(
            "The customer is unlikely to purchase the "
            "Wellness Tourism Package."
        )

    # -----------------------------------------------------
    # Display probability
    # -----------------------------------------------------

    if probability is not None:

        st.metric(
            "Purchase Probability",
            f"{probability:.2%}"
        )