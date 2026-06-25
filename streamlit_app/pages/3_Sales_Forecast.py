import streamlit as st
import pandas as pd
import plotly.express as px
from utils import get_engine

engine = get_engine()

st.header("Sales Forecast: Actual vs Predicted")

stores = pd.read_sql("SELECT DISTINCT store FROM predictions ORDER BY store", engine)
models = pd.read_sql("SELECT DISTINCT model_name FROM predictions", engine)

col1, col2 = st.columns(2)
with col1:
    selected_store = st.selectbox("Select Store", stores['store'])
with col2:
    selected_model = st.selectbox("Select Model", models['model_name'])

query = """
    SELECT sales_date, actual_sales, predicted_sales
    FROM predictions
    WHERE store = %(store)s AND model_name = %(model)s
    ORDER BY sales_date
"""
data = pd.read_sql(query, engine, params={'store': int(selected_store), 'model': selected_model})

fig = px.line(
    data, 
    x='sales_date', 
    y=['actual_sales', 'predicted_sales'],
    title=f"Store {selected_store} — {selected_model}",
    labels={'value': 'Sales', 'sales_date': 'Date'}
)
st.plotly_chart(fig)