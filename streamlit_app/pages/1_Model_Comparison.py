import streamlit as st
import pandas as pd
import plotly.express as px
from utils import get_engine
from utils import apply_custom_theme

engine = get_engine()
# In your model comparison page file
apply_custom_theme(
    main_bg_color="#F2F5F3",     
    sidebar_bg_color="#E3EAE6",  
    text_color="#1E2E24"         
)

st.header("Latest Model Comparison")

query = """
    SELECT model_name, rmse, mae, r2, data_up_to
    FROM model_metrics
    WHERE data_up_to = (SELECT MAX(data_up_to) FROM model_metrics)
    ORDER BY rmse ASC
"""
latest = pd.read_sql(query, engine)
st.dataframe(latest)

fig = px.bar(latest, x='model_name', y='rmse', title='RMSE by Model (Latest Run)', color='model_name')
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color="#2D2544"
)
st.plotly_chart(fig)