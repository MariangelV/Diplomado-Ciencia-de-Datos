import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo guardado
model = joblib.load('modelo_ingresos_organizaciones.pkl')

# Título de la aplicación
st.title("Predicción de Ingresos para Organizaciones sin Fines de Lucro")

st.write("""
Esta aplicación predice los ingresos de una organización en función de varios factores financieros.
Ingresa los datos de tu organización para obtener una estimación de ingresos.
""")

# Crear entradas de usuario para cada variable del modelo
fscore = st.number_input("Puntaje Financiero", min_value=0.0, max_value=100.0, value=50.0)
score = st.number_input("Puntaje de Responsabilidad", min_value=0.0, max_value=100.0, value=50.0)
program_exp = st.number_input("Gastos en Programas", min_value=0.0, value=50000.0)
fund_exp = st.number_input("Gastos en Recaudación de Fondos", min_value=0.0, value=10000.0)
admin_exp = st.number_input("Gastos Administrativos", min_value=0.0, value=15000.0)

# Crear un botón de predicción
if st.button("Predecir Ingresos"):
    # Crear un DataFrame con los valores ingresados por el usuario
    nueva_org = pd.DataFrame({
        'fscore': [fscore],
        'score': [score],
        'program_exp': [program_exp],
        'fund_exp': [fund_exp],
        'admin_exp': [admin_exp]
    })

    # Hacer la predicción
    prediccion_ingresos = model.predict(nueva_org)
    
    # Mostrar el resultado
    st.write(f"Predicción de ingresos para la organización: ${prediccion_ingresos[0]:,.2f}")
