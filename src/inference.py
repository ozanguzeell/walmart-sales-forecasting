import pandas as pd
import pickle

with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

sample = {
    'Store': [1],
    'Dept': [1],
    'IsHoliday': [0],
    'Size': [151315],
    'Temperature': [45.0],
    'Fuel_Price': [3.5],
    'MarkDown1': [0],
    'MarkDown2': [0],
    'MarkDown3': [0],
    'MarkDown4': [0],
    'MarkDown5': [0],
    'CPI': [211.095357],
    'Unemployment': [8.106],
    'Type': [0],
    'Year': [2012],
    'Month': [5],
    'Week': [20],
    'Quarter': [2],
    'DayOfYear': [140]
}

sample_df = pd.DataFrame(sample)

prediction = model.predict(sample_df)

print("Tahmin Edilen Haftalık Satış: ", prediction[0])
