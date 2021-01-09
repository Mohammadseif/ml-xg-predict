import pickle

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor

# Sidebar

st.sidebar.header("متغیرها را وارد کنید")

def user_input_features():
    LF = st.sidebar.slider("LF", 0, 20, 0)
    LM = st.sidebar.slider("LM", 0, 20, 0)
    WM = st.sidebar.slider("WM", 0, 1, 0)
    WL = st.sidebar.slider("WL", 0, 1, 0)
    RLW = st.sidebar.slider("RLW", 0, 25, 0)
    FE = st.sidebar.slider("FE", 0, 1, 0)

    data = {
            'LF':LF,
            'LM':LM,
            'WM':WM,
            'WL':WL,
            'RLW':RLW,
            'FE':RLW,
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

st.header("Orientation Prediction")
st.subheader(" پیش بینی جهت داری ")
st.write("\n")
st.write("\n")
st.write("You have selected the following inputs:")
st.write(df)

with open("xgboost.pkl", "rb") as f:
    load_clf = pickle.load(f)   
predictions = load_clf.predict(df)[0]

st.write("\n")
st.write("\n")
st.subheader("Prediction by XgBoost Model")
st.write(f"The predicted number of bikes today is: {int(predictions)}")

