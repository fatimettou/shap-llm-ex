import streamlit as st
from streamlit_option_menu import option_menu

# Import individual page modules
from my_pages.upload_model import upload_model_page
from my_pages.upload_dataset import upload_dataset_page
from my_pages.generate_shap import generate_shap_page
from my_pages.predictions import predictions_page
from my_pages.bot_app import chatbot_page


# Set up the color scheme using the colors from the Sorbonne logo
st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5;}  /* Light grey background */
    .sidebar .sidebar-content {
        background-color: #1a2d50;  /* Matching deep blue */
        padding: 10px;
    }
    .stButton>button {
        background-color: #d4af37;  /* Gold */
        border-radius: 8px;
        color: white;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #c99a2e;  /* Hover effect for buttons */
    }
    .css-1d391kg {
        font-size: 20px;  /* Larger font for section titles */
        font-weight: bold;
    }
    .chatbox {
        background-color: white;
        border: 2px solid #d4af37;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .chat-message {
        background-color: #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar Navigation with Modern Option Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Upload Model", "Upload Dataset", "Generate SHAP", "Predictions", "Chatbot"],
        icons=["cloud-upload", "file-earmark-spreadsheet", "graph-up", "play", "chat-left-dots"],
        menu_icon="cast",  # Optional icon for the menu
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#1a2d50"},
            "icon": {"color": "#d4af37", "font-size": "20px"},  # Gold for icons
            "nav-link": {"color": "white", "font-size": "18px", "margin": "0px"},
            "nav-link-selected": {"background-color": "#d4af37"},
        },
    )

# Global variables to store model and dataset
if "model" not in st.session_state:
    st.session_state.model = None

if "df" not in st.session_state:
    st.session_state.df = None

# Main Content Based on Selected Page
if selected == "Upload Model":
    model = upload_model_page()
    if model:
        st.session_state.model = model
        st.success("Model is successfully loaded.")

elif selected == "Upload Dataset":
    if st.session_state.model is None:
        st.warning("Please upload the model first.")
    else:
        df = upload_dataset_page()
        if df is not None:
            st.session_state.df = df
            st.success("Dataset uploaded successfully.")

elif selected == "Generate SHAP":
    if st.session_state.model is None:
        st.warning("Please upload the model first.")
    elif st.session_state.df is None:
        st.warning("Please upload the dataset first.")
    else:
        generate_shap_page(st.session_state.model, st.session_state.df)

elif selected == "Predictions":
    if st.session_state.model is None:
        st.warning("Please upload the model first.")
    elif st.session_state.df is None:
        st.warning("Please upload the dataset first.")
    else:
        predictions_page(st.session_state.model)

elif selected == "Chatbot":
    chatbot_page()