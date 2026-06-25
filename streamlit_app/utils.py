import streamlit as st
from sqlalchemy import create_engine
import os

@st.cache_resource
def get_engine():
    db_user = os.environ['DB_USER']
    db_password = os.environ['DB_PASSWORD']
    db_host = os.environ['DB_HOST']
    db_port = os.environ['DB_PORT']
    db_name = os.environ['DB_NAME']
    conn_str = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(conn_str)

def apply_custom_theme(main_bg_color, sidebar_bg_color, text_color="#1E293B", primary_color="#6366F1"):
    st.markdown(
        f"""
        <style>
        /* Make the header transparent and blend into the app */
        header[data-testid="stHeader"] {{
            background-color: rgba(0,0,0,0) !important;
        }}
        
        /* Main App Light Background */
        .stApp {{
            background-color: {main_bg_color};
        }}
        
        /* Sidebar Light Background */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg_color} !important;
        }}
        
        /* Enforce professional, high-contrast typography */
        .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp li, section[data-testid="stSidebar"] * {{
            color: {text_color} !important;
        }}
        
        /* Style widgets nicely for light mode so they don't look broken */
        .stDataFrame, div[data-testid="stMetricValue"] {{
            background-color: #FFFFFF !important;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )