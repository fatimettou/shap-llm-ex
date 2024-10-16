import streamlit as st
import pandas as pd

def upload_dataset_page():
    st.title("Upload Your Dataset")
    
    data_file = st.file_uploader("Upload your dataset CSV file", type=["csv"])
    
    if data_file is not None:
        df = pd.read_csv(data_file)
        st.write("Data Preview:")
        st.write(df.head())
        return df
    return None
