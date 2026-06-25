import streamlit as st
import pandas as pd
import plotly.express as px
from utils import get_engine
from utils import apply_custom_theme

engine = get_engine()
apply_custom_theme(
    main_bg_color="#F1F5F9",     #
    sidebar_bg_color="#E2E8F0", 
    text_color="#0F172A"  )
    
st.header("Metrics Over Time")

history_query = "SELECT model_name, rmse, data_up_to FROM model_metrics ORDER BY data_up_to"
history = pd.read_sql(history_query, engine)

fig2 = px.line(history, x='data_up_to', y='rmse', color='model_name', title='RMSE as Training Data Grows', markers=True)
st.plotly_chart(fig2)