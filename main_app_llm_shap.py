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

# Main Content Based on Selected Page
if selected == "Upload Model":
    upload_model_page()

elif selected == "Upload Dataset":
    upload_dataset_page()

elif selected == "Generate SHAP":
    generate_shap_page()

elif selected == "Predictions":
    predictions_page()

elif selected == "Chatbot":
    chatbot_page()
