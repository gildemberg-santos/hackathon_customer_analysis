""" Importação de bliblitecas para transformação de dados e criação do modelo """
import pandas as pd
from src.predict import Predict

DATA = pd.read_csv("./data/exported_data.csv")
MODELO = './data/modelo'
X = [
    'sale_amount', 'sale_vehicle_id','vehicle_year','vehicle_price', 'lead_bank_score', 
    'lead_have_children', 'lead_years_employment', 'lead_income'
]
Y = 'sale_sold'
modelo_pkl = Predict(MODELO, DATA, X, Y)
modelo_pkl.save()
modelo_pkl.load()
print(modelo_pkl.accuracy)
print(modelo_pkl.y_pred)
