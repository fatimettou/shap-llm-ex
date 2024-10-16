import streamlit as st
import shap
import pandas as pd
import xgboost as xgb
import streamlit.components.v1 as components
from xgb_process import shap_summary  # Assuming ShapAnalyzer is defined in shap_summary module
import os

# Utility function to display SHAP plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def generate_shap_page(model, df):
    if model is None:
        st.warning("Please upload a model.")
        return

    if df is None:
        st.warning("Please upload a dataset.")
        return

    # Ensure the correct column order
    correct_order = ['gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice',
                     'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
                     'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
                     'contract', 'paperlessbilling', 'paymentmethod', 'tenure',
                     'monthlycharges', 'totalcharges']

    cat_cols = ['gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice', 
                'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup', 
                'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
                'contract', 'paperlessbilling', 'paymentmethod']

    num_cols = ['tenure', 'monthlycharges', 'totalcharges']

    # Convert object columns to categorical type
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    target_col = st.selectbox("Select the target column (label)", df.columns)

    if st.button("Generate SHAP Explanations"):
        X = df.drop(columns=[target_col, 'customerid'], errors='ignore')
        X = X[[col for col in correct_order if col in X.columns]]  # Ensure only correct features are used

        y = df[target_col]

        # Ensure all categorical columns are converted to 'category' type
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')

        # Prepare DMatrix for XGBoost with enable_categorical=True
        dmatrix = xgb.DMatrix(X, label=y, enable_categorical=True)

        # Generate SHAP Analysis
        analyzer = shap_summary.ShapAnalyzer(
            model=model,
            X_train=X,
            dtrain=dmatrix,
            cat_features=cat_cols,
            num_features=num_cols
        )

        # Get SHAP values
        shap_data = pd.DataFrame(analyzer.get_shap_value(), columns=X.columns)

        st.write(f"The local SHAP explanation for test data is {shap_data.shape}")

        # Call analyze_shap_values to make sure that result_df is generated
        result_df = analyzer.analyze_shap_values()

        if result_df is None:
            st.error("Error: SHAP result dataframe is empty or not properly generated.")
            return

        # Save the local explanation summary as a text file
        summary_text = analyzer.summarize_shap_text()

        if summary_text is None:
            st.error("Error in generating SHAP text summary. No SHAP values found.")
            return

        # Ensure the 'documents' directory exists
        save_path = "documents"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = os.path.join(save_path, 'shap_summary.txt')
        with open(file_path, 'w') as f:
            f.write("SHAP contribution:\n")
            f.write(summary_text)

        st.success(f"Local Feature Contribution Summary saved to: {file_path}")
        
        # Download button for the SHAP summary file
        with open(file_path, 'r') as f:
            file_contents = f.read()

        st.download_button(
            label="Download SHAP Summary",
            data=file_contents,
            file_name="shap_summary.txt",
            mime="text/plain"
        )

        # Display SHAP summary plot
        st.subheader("SHAP Summary Plot")
        shap.summary_plot(shap_data.values, X, plot_type="bar", show=False)
        st.pyplot(bbox_inches='tight')

        # SHAP Force Plot for the first prediction
        st.subheader("SHAP Force Plot for First Prediction")
        st_shap(shap.force_plot(analyzer.explainer.expected_value, shap_data.values[0, :], X.iloc[0, :]))
