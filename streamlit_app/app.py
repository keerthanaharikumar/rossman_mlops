import streamlit as st
from utils import apply_custom_theme

st.set_page_config(page_title="Rossmann Sales Forecasting", layout="wide")
apply_custom_theme(
    main_bg_color="#F4F2FA", 
    sidebar_bg_color="#E8E3F5", 
    text_color="#2D2544"
)

st.title("Rossmann Sales Forecasting")
st.write("""
This dashboard tracks an end-to-end MLOps pipeline for forecasting Rossmann store sales.
Built with Airflow, Postgres, and three competing ML models.
         
There are two DAG defined for ingestion of data and for training.
You can see how the results differ with each trigger.

Use the sidebar to navigate between pages:
- **Model Comparison** :latest run results across all three models
- **Metrics Over Time** :how performance improved as more data was ingested
- **Sales Forecast** : store-level predictions vs actuals
""")