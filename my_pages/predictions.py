import streamlit as st
import xgboost as xgb
import pandas as pd

def predictions_page(model, df, correct_order):
    if model is None:
        st.warning("Please upload a model.")
        return

    if df is None:
        st.warning("Please upload a dataset.")
        return

    # Convert object columns to categorical types
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    # Let the user select the target column for prediction
    target_col = st.selectbox("Select the target column for prediction", df.columns)

    if st.button("Run Predictions"):
        X = df.drop(columns=[target_col, 'customerid'], errors='ignore')
        y = df[target_col]

        # Ensure columns are in the correct order
        missing_columns = set(correct_order) - set(X.columns)
        if missing_columns:
            st.error(f"Missing columns for prediction: {missing_columns}")
            return

        # Reorder the dataset to match the correct column order (used in training)
        X = X[correct_order]

        # Prepare DMatrix for XGBoost
        dmatrix = xgb.DMatrix(X, enable_categorical=True)

        # Generate predictions
        st.write("Running predictions...")
        predictions = model.predict(dmatrix)

        # If binary classification, apply a threshold for Yes/No or 1/0 classification
        threshold = 0.5
        predicted_classes = (predictions > threshold).astype(int)
        predicted_labels = ["Yes" if pred == 1 else "No" for pred in predicted_classes]

        # Display predictions in a table
        result_df = X.copy()
        result_df["Actual"] = y
        result_df["Predicted Probability"] = predictions
        result_df["Predicted Class"] = predicted_labels
        st.write(result_df)

        st.success("Predictions complete!")
