import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def main():
    st.title('Telco Customer Churn Prediction')
    img=Image.open(BASE_DIR / "image.png")
    st.image(img,width=500)

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

    # Add any other feature columns required by your trained model.


    if st.button("Predict"):
        feature = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "InternetService": InternetService,
            "PaymentMethod": PaymentMethod,
            "Contract": Contract,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])

        with open(BASE_DIR / "random_forest_model.pkl", "rb") as f:
            rf = pickle.load(f)

        with open(BASE_DIR / "encoded_columns.pkl", "rb") as f:
            columns = pickle.load(f)


        # Remove target column from saved features if it exists
        columns = [col for col in columns if col != "Churn"]

        # Encode categorical inputs
        feature = pd.get_dummies(feature)

        # Match the training columns
        feature = feature.reindex(columns=columns, fill_value=0)

        # # Verify before prediction
        # print("Churn in input:", "Churn" in feature.columns)
        # print("Input shape:", feature.shape)
        # print("Expected feature count:", rf.n_features_in_)

        # Predict
        result = int(rf.predict(feature)[0])

        if result == 0:
            st.success("Customer is likely to stay(Churn:No)")
        else:
            st.warning("Customer is likely to churn(Churn:Yes)")

main()