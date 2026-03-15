import streamlit as st
import joblib
import numpy as np

model = joblib.load("insurance_model.pkl")

st.title("Insurance Cost Prediction")

age = st.number_input("Age")
sex = st.selectbox("Sex",[0,1])
bmi = st.number_input("BMI")
children = st.number_input("Children")
smoker = st.selectbox("Smoker",[0,1])
region_northwest = st.selectbox("Region Northwest",[0,1])
region_southeast = st.selectbox("Region Southeast",[0,1])
region_southwest = st.selectbox("Region Southwest",[0,1])

data = np.array([[age,sex,bmi,children,smoker,
region_northwest,region_southeast,region_southwest]])

if st.button("Predict"):
    prediction = model.predict(data)
    st.success(f"Predicted Cost: {prediction[0]}")