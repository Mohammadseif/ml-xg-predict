import pickle

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from sklearn.externals import joblib

# Sidebar

st.sidebar.header("متغیرها را وارد کنید")

def user_input_features():
    LF = st.sidebar.slider("LF", 0.00, 20.00, 0.00)
    LM = st.sidebar.slider("LM", 0.00, 20.00, 0.0)
    WM = st.sidebar.slider("WM", 0.00, 1.00, 0.00)
    WL = st.sidebar.slider("WL", 0.00, 1.00, 0.00)
    RLW = st.sidebar.slider("RLW", 0.00, 25.00, 0.00)
    FE = st.sidebar.slider("FE", 0.00, 1.00, 0.00)

    data = {
            'LF':LF,
            'LM':LM,
            'WM':WM,
            'WL':WL,
            'RLW':RLW,
            'FE':FE,
    }

    return pd.DataFrame(data, index=[0])    

df = user_input_features()
st.sidebar.markdown(
""" 
****   
[Download training data](https://raw.githubusercontent.com/cambridgecoding/machinelearningregression/master/data/bikes.csv)
"""
)


# Main

st.header("شرکت بهینه راهبرد انفجار***")
st.subheader(" پیش بینی جهت داری ")
st.write("\n")
st.write("\n")
st.write("متغیرهای ورودی انتخاب کرده اید:")
st.write(df)

with open("xgboost.pkl", "rb") as f:
    mdl = joblib.load(f)   
predictions = mdl.predict(df)[0]

st.write("\n")
st.write("\n")
st.subheader("Prediction by XgBoost Model")
st.write(f"The predicted number of bikes today is: {(predictions)}")

