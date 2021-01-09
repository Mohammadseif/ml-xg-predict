import pickle

import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
import xlrd

print("Downloading data...")
df = pd.read_excel('data2.xlsx')
data= df.loc[df['v7'] == 'magnetit']

xgb_data = df1.copy()

y_df = xgb_data['Orientation'].reset_index(drop=True)
x_df = xgb_data[['LF', 'LM', 'WM', 'WL', 'RLW', 'FE']]

train_x, test_x, train_y, test_y = train_test_split(x_df, y_df, test_size=0.20,random_state = 42)
print("Training model...")                                                   

model = XGBClassifier(
    learning_rate = 0.00599657208814288 , 
    n_estimators = 1000 , 
    max_depth =6 , 
    min_child_weight =1.0 , 
    gamma =0.9 , 
    subsample = 0.9500000000000001, 
    colsample_bytree = 0.6000000000000001 ,
    objective = 'binary:logistic')
model.fit(train_x, train_y)   
predictions = model.predict(test_x)

score = r2_score(test_y, predictions)
print(f"R2 score on test-set is {score}")

print("Saving model...")
with open("xgboost.pkl", "wb") as f:
    pickle.dump(model,f)
print("xgboost.pkl")
