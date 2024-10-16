import streamlit as st
import tempfile
import xgboost as xgb

def upload_model_page():
    st.title("Upload Your Model")
    
    model_file = st.file_uploader("Upload your pre-trained XGBoost model (JSON format)", type=["json"])
    
    if model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(model_file.read())
            temp_file.flush()
            model = xgb.Booster()
            model.load_model(temp_file.name)

        return model
    return None
