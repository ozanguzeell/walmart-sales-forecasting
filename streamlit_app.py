import streamlit as st
import pandas as pd
import pickle

# Load Model
with open('../models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("📈 Walmart Sales Forecasting App")

st.markdown("Store, Dept ve bazı özellikleri girerek haftalık satış tahmini yapın.")

store = st.number_input("Store", min_value=1, max_value=45, step=1)
dept = st.number_input("Department", min_value=1, max_value=99, step=1)
size = st.number_input("Store Size", min_value=0)
year = st.number_input("Year", min_value=2010, max_value=2013)
month = st.number_input("Month", min_value=1, max_value=12)
week = st.number_input("Week", min_value=1, max_value=52)
isholiday = st.selectbox("Holiday?", [0, 1])

data = pd.DataFrame({
    'Store': [store],
    'Dept': [dept],
    'Size': [size],
    'Year': [year],
    'Month': [month],
    'Week': [week],
    'IsHoliday': [isholiday]
})

if st.button("Predict Sales"):
    prediction = model.predict(data)[0]
    st.success(f"📌 Tahmini Haftalık Satış: **${prediction:,.2f}**")
