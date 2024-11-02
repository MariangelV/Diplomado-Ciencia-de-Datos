import joblib
from sklearn.neural_network import MLPRegressor
import pandas as pd

# Datos de ejemplo (X y y), reemplaza con tus datos reales
X = pd.DataFrame({
    'fscore': [80, 70, 60, 90],
    'score': [85, 75, 65, 95],
    'program_exp': [50000, 60000, 70000, 80000],
    'fund_exp': [10000, 15000, 12000, 20000],
    'admin_exp': [15000, 16000, 14000, 18000]
})
y = [200000, 250000, 230000, 300000]  # Reemplaza con los ingresos reales

# Entrena y guarda el modelo
model = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
model.fit(X, y)
joblib.dump(model, 'modelo_ingresos_organizaciones.pkl')
