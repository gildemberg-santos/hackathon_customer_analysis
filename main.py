import pandas as pd
from src.predict import Predict

data = pd.read_csv("./data/exported_data.csv")
modelo = './data/modelo'
x = ['sale_amount', 'sale_vehicle_id','vehicle_year','vehicle_price', 'lead_bank_score', 'lead_have_children', 'lead_years_employment', 'lead_income']
y = 'sale_sold'
modelo_pkl = Predict(modelo, data, x, y)
modelo_pkl.save()
modelo_pkl.load()
print(modelo_pkl.accuracy)
print(modelo_pkl.y_pred)