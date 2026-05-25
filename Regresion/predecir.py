import h5py
import pandas as pd

with h5py.File("modelo_pesos.h5", "r") as f:
    coeficientes = f["coeficientes"][:]
    intercepto = f["intercepto"][:][0]

print("--- SISTEMA DE PREDICCION DE VENTAS (REGRESION) ---\n")
try:
    publicidad = float(input("Gasto en publicidad: "))
    precio = float(input("Precio del producto: "))
except ValueError:
    print("[ERROR] Ingresa valores numericos validos.")
    exit()

prediccion = (coeficientes[0] * publicidad) + (coeficientes[1] * precio) + intercepto
print(f"\n-> Volumen estimado de ventas: {max(0, prediccion):.2f} unidades.")
