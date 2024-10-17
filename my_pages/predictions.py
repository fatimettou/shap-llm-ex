import streamlit as st
import pandas as pd
import xgboost as xgb

# Update the function to accept df and correct_order as arguments
def predictions_page(model):
    st.title("Make Predictions")

    st.subheader("Enter the features for the prediction:")

    # Use columns to organize input features
    col1, col2, col3 = st.columns(3)
    correct_order = ['gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice',
                 'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
                 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
                 'contract', 'paperlessbilling', 'paymentmethod', 'tenure',
                 'monthlycharges', 'totalcharges']
    with col1:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        seniorcitizen = st.selectbox("Senior Citizen", options=[0, 1])
        partner = st.selectbox("Partner", options=["Yes", "No"])
        dependents = st.selectbox("Dependents", options=["Yes", "No"])
        phoneservice = st.selectbox("Phone Service", options=["Yes", "No"])

    with col2:
        multiplelines = st.selectbox("Multiple Lines", options=["Yes", "No", "No phone service"])
        internetservice = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
        onlinesecurity = st.selectbox("Online Security", options=["Yes", "No", "No internet service"])
        onlinebackup = st.selectbox("Online Backup", options=["Yes", "No", "No internet service"])
        deviceprotection = st.selectbox("Device Protection", options=["Yes", "No", "No internet service"])

    with col3:
        techsupport = st.selectbox("Tech Support", options=["Yes", "No", "No internet service"])
        streamingtv = st.selectbox("Streaming TV", options=["Yes", "No", "No internet service"])
        streamingmovies = st.selectbox("Streaming Movies", options=["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
        paperlessbilling = st.selectbox("Paperless Billing", options=["Yes", "No"])
        paymentmethod = st.selectbox("Payment Method", options=[
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    # Additional inputs for numerical fields
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure", 0, 72, 12)
        monthlycharges = st.slider("Monthly Charges", 0.0, 120.0, 50.0)

    with col2:
        totalcharges = st.slider("Total Charges", 0.0, 9000.0, 1000.0)

    # Once user inputs are ready, make predictions
    if st.button("Run Predictions"):
        # Prepare data for prediction
        input_data = {
            'gender': gender,
            'seniorcitizen': seniorcitizen,
            'partner': partner,
            'dependents': dependents,
            'phoneservice': phoneservice,
            'multiplelines': multiplelines,
            'internetservice': internetservice,
            'onlinesecurity': onlinesecurity,
            'onlinebackup': onlinebackup,
            'deviceprotection': deviceprotection,
            'techsupport': techsupport,
            'streamingtv': streamingtv,
            'streamingmovies': streamingmovies,
            'contract': contract,
            'paperlessbilling': paperlessbilling,
            'paymentmethod': paymentmethod,
            'tenure': tenure,
            'monthlycharges': monthlycharges,
            'totalcharges': totalcharges
        }

        # Convert input data into DataFrame format
        input_df = pd.DataFrame([input_data])

        # Ensure the column order matches the correct_order
        input_df = input_df[correct_order]

        # Convert categorical columns to 'category' dtype
        categorical_columns = input_df.select_dtypes(include=['object']).columns
        input_df[categorical_columns] = input_df[categorical_columns].astype('category')

        # Ensure the use of enable_categorical=True in DMatrix
        dmatrix = xgb.DMatrix(input_df, enable_categorical=True)
        
        # Make prediction
        prediction_proba = model.predict(dmatrix)

        # Apply threshold to convert probabilities to 0 or 1
        prediction = (prediction_proba > 0.5).astype(int)

        # Display prediction result as Yes/No
        prediction_label = "Yes" if prediction[0] == 1 else "No"

        st.write("Prediction Result:")
        st.write(f"Churn: {prediction_label}")
