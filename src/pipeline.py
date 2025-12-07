import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pickle

train = pd.read_csv('data/train.csv')
stores = pd.read_csv('data/stores.csv')
features = pd.read_csv('data/features.csv')

train['Date'] = pd.to_datetime(train['Date'])
features['Date'] = pd.to_datetime(features['Date'])

data = train.merge(stores, on='Store', how='left')
data = data.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')

markdown_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
data[markdown_cols] = data[markdown_cols].fillna(0)

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.isocalendar().week.astype(int)
data['Quarter'] = data['Date'].dt.quarter
data['DayOfYear'] = data['Date'].dt.dayofyear

le = LabelEncoder()
data['Type'] = le.fit_transform(data['Type'])

y = data['Weekly_Sales']
X = data.drop(['Weekly_Sales','Date'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model başarıyla eğitildi ve models/xgboost_model.pkl olarak kaydedildi!")
