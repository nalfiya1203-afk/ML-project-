import os
import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# ---------------- CSS ----------------
CSS = """
<style>
/* page width and spacing */
.block-container {
    max-width: 760px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* title */
h1 {
    text-align: center;
    color: #4f46e5;
    font-weight: 800;
    letter-spacing: 0.5px;
}

/* image: centered with rounded corners */
[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
}
[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    margin-bottom: 1rem;
}

/* widget labels */
label, [data-testid="stWidgetLabel"] p {
    font-weight: 600 !important;
}

/* dropdowns and number boxes */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
    border: 1px solid #a5b4fc;
}

/* predict button */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    font-size: 18px;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1rem;
    margin-top: 1rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(79, 70, 229, 0.45);
    color: white;
}

/* result boxes */
[data-testid="stAlert"] {
    border-radius: 12px;
    font-size: 17px;
    font-weight: 600;
}
</style>
"""


def main():
    st.markdown(CSS, unsafe_allow_html=True)

    st.title('Telco Customer Churn Prediction')
    if os.path.exists('img.png'):
        img = Image.open('img.png')
        st.image(img, width=500)

    gender = st.selectbox("Select Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                    "Bank transfer (automatic)", "Credit card (automatic)"])
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=500.0)

    # The remaining feature columns required by the trained model
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

    if st.button("Predict"):
        # Dataset rules: no internet -> these services are "No internet service",
        # no phone -> MultipleLines is "No phone service"
        if InternetService == "No":
            OnlineSecurity = OnlineBackup = DeviceProtection = "No internet service"
            TechSupport = StreamingTV = StreamingMovies = "No internet service"
        if PhoneService == "No":
            MultipleLines = "No phone service"

        feature = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])

        with open("random_forest_model.pkl", "rb") as f:
            rf = pickle.load(f)

        with open("encoded_columns.pkl", "rb") as f:
            columns = pickle.load(f)

        # Remove target column from saved features if it exists
        columns = [col for col in columns if col != "Churn"]

        # Safety check: saved columns must match the model
        if list(rf.feature_names_in_) != columns:
            st.error("encoded_columns.pkl does not match random_forest_model.pkl. "
                     "Run the notebook again and save both files together.")
            st.stop()

        # Encode categorical inputs
        feature = pd.get_dummies(feature, dtype=int)

        # Match the training columns
        feature = feature.reindex(columns=columns, fill_value=0)

        # Predict
        result = int(rf.predict(feature)[0])
        prob = float(rf.predict_proba(feature)[0][1])

        if result == 0:
            st.success(f"Customer is likely to stay (Churn: No) - churn probability {prob:.0%}")
        else:
            st.warning(f"Customer is likely to churn (Churn: Yes) - churn probability {prob:.0%}")


main()