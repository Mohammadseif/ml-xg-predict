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
    LF = st.sidebar.slider("میانگین عمق چال‌ها در ردیف اول", 0.00, 20.00, 0.00)
    LM = st.sidebar.slider("میانگین عمق چال‌ها در ردیف‌های میانی", 0.00, 20.00, 0.0)
    WM = st.sidebar.slider("نسبت عمق آب به عمق چال ردیف‌های میانی", 0.00, 1.00, 0.00)
    WL = st.sidebar.slider("نسبت عمق آب به عمق چال ردیف آخر", 0.00, 1.00, 0.00)
    RLW = st.sidebar.slider("نسبت طول به عرض بلوک  انفجاری", 0.00, 25.00, 0.00)

    data = {
            'LF':LF,
            'LM':LM,
            'WM':WM,
            'WL':WL,
            'RLW':RLW,
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

st.header("***Mine to Mill Optimization Project in Goharzamin Iron Ore Mine***")
st.subheader(" پیش بینی کارایی خردایش ")
st.write("\n")
st.write("\n")
st.write("متغیرهای ورودی کاربر:")
st.write(df)

with open("xgboost.pkl", "rb") as f:
    mdl = joblib.load(f)   
predictions = mdl.predict(df)[0]

st.write("\n")
st.write("\n")
st.subheader("پیش بینی بر اساس مدل : XgBoost")
st.write(f"The predicted Fragmentation is: {(predictions)}")

