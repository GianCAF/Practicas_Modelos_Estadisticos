import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import h5py
import yaml

# =====================================================================
# 1. ANALIZAR EL CONJUNTO DE DATOS 
# =====================================================================
try:
    data = pd.read_csv("datos_regresion.csv")
except FileNotFoundError:
    print("[ERROR] No se encontro 'datos_regresion.csv'.")
    exit()

print("--- ANALISIS DEL CONJUNTO DE DATOS ---")
print(f"Dimensiones del CSV: {data.shape}")

# Variables para nuestra Regresion Lineal Multiple
X = data[["publicidad", "precio"]]
y = data["ventas"]

# =====================================================================
# 2. PARTICIÓN DE DATOS 
# =====================================================================
# Dividimos en 80% entrenamiento y 20% prueba para evitar sobreajuste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("-> Particion de datos realizada con exito.")

# =====================================================================
# 3. CROSS-VALIDATION / AJUSTE DE VARIABLES 
# =====================================================================
modelo = LinearRegression()

# Evaluamos la estabilidad del modelo con Cross-Validation (3 pliegues)
# En regresion se mide con el Error Cuadratico Medio Negativo (NMSE)
cv_scores = cross_val_score(modelo, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-cv_scores.mean())

# =====================================================================
# 4. ENTRENAMIENTO DEL MODELO DE REGRESIÓN 
# =====================================================================
modelo.fit(X_train, y_train)
print("--- Modelo entrenado con exito ---")

# =====================================================================
# 5. METRICAS DE EVALUACION / ACCURACY 
# =====================================================================
# Hacemos las predicciones con el conjunto de prueba (Test)
predicciones = modelo.predict(X_test)

# Como es regresion, el "Accuracy" se mide con R2 Score (Precision del ajuste)
precision_r2 = r2_score(y_test, predicciones)

print("\n--- METRICAS DE EVALUACION DEL MODELO ---")
print(f"-> Promedio Error Cross-Validation (RMSE): {rmse_cv:.4f}")
print(f"-> Precision R2 Score (Equivalente a Accuracy): {precision_r2:.4f} ({precision_r2*100:.1f}%)")

# Mostrar matriz de diferencias (Muestra los valores reales vs predichos)
print("\n--- TABLA DE COMPARACION (EVALUACION REAL VS PREDICCION) ---")
df_eval = pd.DataFrame({'Real': y_test, 'Prediccion': predicciones, 'Diferencia': y_test - predicciones})
print(df_eval.to_string(index=False))

# =====================================================================
# 6. EXPORTACIÓN A LOS ARCHIVOS REQUERIDOS (.h5 y .yml) 
# =====================================================================
# A. Guardar coeficientes e intercepto en formato H5
with h5py.File("modelo_pesos.h5", "w") as f:
    f.create_dataset("coeficientes", data=modelo.coef_)
    f.create_dataset("intercepto", data=np.array([modelo.intercept_]))
print("\n-> Archivo 'modelo_pesos.h5' guardado.")

# B. Guardar la configuracion en YAML
config_dict = {
    "modelo": {
        "tipo": "LinearRegression",
        "variables_independientes": ["publicidad", "precio"],
        "variable_dependiente": "ventas",
        "libreria": "scikit-learn"
    }
}
with open("config.yml", "w") as f:
    yaml.dump(config_dict, f, default_flow_style=False)
print("-> Archivo 'config.yml' guardado.")

# C. Generar el script automatizado .py para predicciones individuales
script_codigo = """import h5py
import pandas as pd

with h5py.File("modelo_pesos.h5", "r") as f:
    coeficientes = f["coeficientes"][:]
    intercepto = f["intercepto"][:][0]

print("--- SISTEMA DE PREDICCION DE VENTAS (REGRESION) ---\\n")
try:
    publicidad = float(input("Gasto en publicidad: "))
    precio = float(input("Precio del producto: "))
except ValueError:
    print("[ERROR] Ingresa valores numericos validos.")
    exit()

prediccion = (coeficientes[0] * publicidad) + (coeficientes[1] * precio) + intercepto
print(f"\\n-> Volumen estimado de ventas: {max(0, prediccion):.2f} unidades.")
"""
with open("predecir.py", "w", encoding="utf-8") as f:
    f.write(script_codigo)
print("-> Script 'predecir.py' generado.")