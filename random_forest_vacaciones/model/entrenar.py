import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import joblib

# Suprimir warnings informativos de sklearn sobre clases únicas
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,        # ← [D] necesario para la tabla de confusión
    ConfusionMatrixDisplay,  # ← [D] visualización opcional en consola
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_VARS  = os.path.join(BASE_DIR, "config", "variables.yml")
CFG_LIMS  = os.path.join(BASE_DIR, "config", "limites.yml")
CSV_PATH  = os.path.join(BASE_DIR, "data",   "destinos_mexico_100.csv")
MODEL_OUT = os.path.join(BASE_DIR, "model",  "modelo_rf.joblib")


# ── 1. Cargar configuración ──────────────────────────────────────────────────
def cargar_config():
    with open(CFG_VARS, "r", encoding="utf-8") as f:
        vars_cfg = yaml.safe_load(f)
    with open(CFG_LIMS, "r", encoding="utf-8") as f:
        lims_cfg = yaml.safe_load(f)
    return vars_cfg, lims_cfg


# ── 2. Parsear columnas multi-etiqueta del CSV ────────────────────────────────
def parsear_multivalue(serie: pd.Series) -> list[list[str]]:
    """'playa|relax|gastronomia'  →  ['playa', 'relax', 'gastronomia']"""
    return [str(v).split("|") for v in serie]


# [B] FUNCIÓN MATEMÁTICA DEL RANDOM FOREST
def construir_features(df: pd.DataFrame, vars_cfg: dict,
                        mlb_clima=None, mlb_gustos=None,
                        scaler=None, fit=True):
    """
    Construye el vector de features X combinando los 3 sub-bosques ponderados.

    [B] La concatenación final implementa la parte matemática de ponderación:
        X = hstack( X_clima·w_c, X_pres·w_p, X_gustos·w_g )
    """

    # --- Sub-bosque CLIMA (MultiLabelBinarizer) ---
    # Convierte ['caluroso','lluvioso'] → [1,0,0,1,0]  (one-hot multilabel)
    climas_lista = parsear_multivalue(df["clima"])
    if fit:
        mlb_clima = MultiLabelBinarizer(
            classes=vars_cfg["features"]["clima"]["valores_posibles"]
        )
        X_clima = mlb_clima.fit_transform(climas_lista)
    else:
        X_clima = mlb_clima.transform(climas_lista)

    # --- Sub-bosque PRESUPUESTO + TIEMPO ---
    # MinMaxScaler normaliza a [0,1]: x' = (x - min) / (max - min)
    X_pres = df[["min_budget", "min_dias"]].values.astype(float)
    if fit:
        scaler = MinMaxScaler()
        X_pres = scaler.fit_transform(X_pres)
    else:
        X_pres = scaler.transform(X_pres)

    # --- Sub-bosque GUSTOS (MultiLabelBinarizer) ---
    # Convierte ['playa','relax'] → [0,0,0,1,0,1,0]  (one-hot multilabel)
    gustos_lista = parsear_multivalue(df["gustos"])
    if fit:
        mlb_gustos = MultiLabelBinarizer(
            classes=vars_cfg["features"]["gustos"]["valores_posibles"]
        )
        X_gustos = mlb_gustos.fit_transform(gustos_lista)
    else:
        X_gustos = mlb_gustos.transform(gustos_lista)

    # [B] APLICACIÓN DE PESOS → implementa X_final = [Xc·wc | Xp·wp | Xg·wg]
    w_clima  = vars_cfg["sub_bosques"]["peso_clima"]               # 0.30
    w_pres   = vars_cfg["sub_bosques"]["peso_presupuesto_tiempo"]  # 0.40
    w_gustos = vars_cfg["sub_bosques"]["peso_gustos"]              # 0.30

    # [B] hstack concatena los tres sub-vectores ya ponderados en una sola fila
    X = np.hstack([
        X_clima  * w_clima,   # [B] columnas del clima  × 0.30
        X_pres   * w_pres,    # [B] columnas de presupuesto+días × 0.40
        X_gustos * w_gustos,  # [B] columnas de gustos × 0.30
    ])

    return X, mlb_clima, mlb_gustos, scaler


# ── 4. Generación de datos sintéticos (20 %) ─────────────────────────────────
def generar_sinteticos(df: pd.DataFrame, lims_cfg: dict,
                        vars_cfg: dict) -> pd.DataFrame:
    """
    Crea datos sintéticos variando presupuesto ±10% y días ±1
    sobre registros reales. Mantiene clima y gustos originales.
    (Este 20% amplía el dataset para mejorar la generalización del bosque)
    """
    rng    = np.random.default_rng(lims_cfg["sinteticos"]["semilla"])
    ratio  = vars_cfg["data"]["synthetic_ratio"]
    n_sint = max(1, int(len(df) * ratio / (1 - ratio)))  # ≈20% del total

    idx_base = rng.choice(len(df), size=n_sint, replace=True)
    df_base  = df.iloc[idx_base].copy().reset_index(drop=True)

    ruido_pct = lims_cfg["sinteticos"]["presupuesto_ruido_pct"]
    ruido     = rng.uniform(-ruido_pct, ruido_pct, size=n_sint)
    df_base["min_budget"] = (
        df_base["min_budget"] * (1 + ruido)
    ).clip(
        lims_cfg["presupuesto"]["minimo"],
        lims_cfg["presupuesto"]["maximo"]
    ).round(-2).astype(int)

    var_dias = lims_cfg["sinteticos"]["dias_variacion"]
    df_base["min_dias"] = (
        df_base["min_dias"] + rng.integers(-var_dias, var_dias + 1, size=n_sint)
    ).clip(
        lims_cfg["dias"]["minimo"],
        lims_cfg["dias"]["maximo"]
    )

    df_base["_origen"] = "sintetico"
    return df_base


# ════════════════════════════════════════════════════════════════════════════
# [A] CROSS-VALIDATION  (Validación Cruzada K-Fold)
# ════════════════════════════════════════════════════════════════════════════
# Objetivo: estimar qué tan bien generaliza el modelo con datos NO vistos,
# sin necesidad de un conjunto de prueba separado.
#
# Funcionamiento con K=5 folds:
#
#   Dataset completo (126 registros)
#   ┌──────┬──────┬──────┬──────┬──────┐
#   │Fold 1│Fold 2│Fold 3│Fold 4│Fold 5│
#   └──────┴──────┴──────┴──────┴──────┘
#
#   Iteración 1: entrena con [2,3,4,5] → evalúa en [1]  → accuracy₁
#   Iteración 2: entrena con [1,3,4,5] → evalúa en [2]  → accuracy₂
#   ...
#   Iteración 5: entrena con [1,2,3,4] → evalúa en [5]  → accuracy₅
#
#   Accuracy_CV = promedio(accuracy₁ … accuracy₅)
#
# Con 80% datos reales + 20% sintéticos = 126 registros totales.
# KFold (no estratificado) porque hay 101 clases únicas (1 por destino).
# ════════════════════════════════════════════════════════════════════════════
def ejecutar_cv(X: np.ndarray, y: np.ndarray, vars_cfg: dict):
    """
    [A] Ejecuta K-Fold Cross-Validation sobre el dataset completo.
    Retorna el clasificador (sin ajustar) para reutilizarlo en el main.
    """
    rf_params = vars_cfg["random_forest"]
    cv_params = vars_cfg["cross_validation"]

    # [A] Se instancia el mismo RandomForest que se usará en producción
    clf = RandomForestClassifier(
        n_estimators      = rf_params["n_estimators"],     # núm. de árboles
        max_depth         = rf_params["max_depth"],         # profundidad máx.
        min_samples_split = rf_params["min_samples_split"], # muestras p/dividir
        min_samples_leaf  = rf_params["min_samples_leaf"],  # muestras en hoja
        max_features      = rf_params["max_features"],      # features por nodo
        random_state      = rf_params["random_state"],      # semilla reproducible
        n_jobs            = rf_params["n_jobs"],            # núcleos CPU
        class_weight      = rf_params["class_weight"],      # balance de clases
    )

    # [A] KFold divide el dataset en n_splits partes sin estratificar
    kf = KFold(
        n_splits     = cv_params["n_splits"],    # K = 5 particiones
        shuffle      = cv_params["shuffle"],     # mezcla antes de dividir
        random_state = cv_params["random_state"],
    )

    # [A] cross_val_score entrena y evalúa K veces → devuelve array de K scores
    scores = cross_val_score(clf, X, y, cv=kf, scoring="accuracy")

    # [A] Reporte de resultados del cross-validation
    print("\n" + "=" * 55)
    print("  [A] CROSS-VALIDATION (K-Fold = {})".format(cv_params["n_splits"]))
    print("=" * 55)
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}  ← accuracy en el fold {i} de prueba")
    print(f"  Promedio : {scores.mean():.4f}  ← accuracy promedio general")
    print(f"  Desv.Est.: {scores.std():.4f}  ← qué tan estables son los folds")
    print("=" * 55 + "\n")
    return clf


# ════════════════════════════════════════════════════════════════════════════
# [C] INICIO DEL MODELO  (Entrenamiento final con todos los datos)
# ════════════════════════════════════════════════════════════════════════════
# Aquí el Random Forest aprende definitivamente.
# Se usa TODO el dataset (80% real + 20% sintético) porque el cross-validation
# ya nos dio la estimación de generalización.
#
# clf.fit(X, y) dispara internamente:
#   1. Para cada árbol i = 1 … n_estimators:
#      a) Bootstrap: selecciona n muestras aleatorias CON reemplazo
#      b) Construye el árbol optimizando Gini en cada nodo
#      c) En cada nodo elige las mejores √p features disponibles
#   2. Guarda los n_estimators árboles entrenados en memoria
# ════════════════════════════════════════════════════════════════════════════
def entrenar_final(X: np.ndarray, y: np.ndarray,
                   vars_cfg: dict) -> RandomForestClassifier:
    """
    [C] Instancia y ajusta el modelo final sobre el dataset completo.
    Este clf.fit() es el INICIO real del aprendizaje del bosque.
    """
    rf_params = vars_cfg["random_forest"]

    # [C] Instancia del RandomForest con todos los hiperparámetros configurados
    clf = RandomForestClassifier(
        n_estimators      = rf_params["n_estimators"],
        max_depth         = rf_params["max_depth"],
        min_samples_split = rf_params["min_samples_split"],
        min_samples_leaf  = rf_params["min_samples_leaf"],
        max_features      = rf_params["max_features"],
        random_state      = rf_params["random_state"],
        n_jobs            = rf_params["n_jobs"],
        class_weight      = rf_params["class_weight"],
    )

    # [C] ★ AQUÍ INICIA EL MODELO ★
    # clf.fit() construye los 200 árboles de decisión usando Gini + Bootstrap
    # A partir de esta línea el objeto 'clf' contiene el bosque entrenado
    clf.fit(X, y)

    return clf


# ── 7. Guardar artefacto .joblib ──────────────────────────────────────────────
def guardar_modelo(clf, mlb_clima, mlb_gustos, scaler,
                   vars_cfg, df_destinos):
    artefacto = {
        "modelo"      : clf,
        "mlb_clima"   : mlb_clima,
        "mlb_gustos"  : mlb_gustos,
        "scaler"      : scaler,
        "vars_cfg"    : vars_cfg,
        "df_destinos" : df_destinos,
    }
    joblib.dump(artefacto, MODEL_OUT)
    print(f"  ✅  Modelo guardado en: {MODEL_OUT}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌴  Random Forest Vacaciones México — Entrenamiento")
    print("─" * 55)

    # --- Carga de config y datos ---
    vars_cfg, lims_cfg = cargar_config()
    print("  ✔  Configuración cargada")

    df = pd.read_csv(CSV_PATH)
    df["_origen"] = "real"
    print(f"  ✔  CSV cargado: {len(df)} destinos reales")

    df_sint = generar_sinteticos(df, lims_cfg, vars_cfg)
    print(f"  ✔  Datos sintéticos generados: {len(df_sint)} registros")

    df_total = pd.concat([df, df_sint], ignore_index=True)
    print(f"  ✔  Dataset total: {len(df_total)} registros "
          f"({len(df)} reales + {len(df_sint)} sintéticos)")

    # [B] Construcción de features con los 3 sub-bosques ponderados
    X, mlb_clima, mlb_gustos, scaler = construir_features(
        df_total, vars_cfg, fit=True
    )
    y = df_total["nombre"].values
    print(f"  ✔  Features construidas: shape {X.shape}  "
          f"← {X.shape[1]} columnas = 4 clima + 2 presup + 7 gustos (ponderados)")

    # [A] Ejecutar Cross-Validation antes del entrenamiento final
    clf_cv = ejecutar_cv(X, y, vars_cfg)

    # ════════════════════════════════════════════════════════════════════════
    # [D] TABLA DE CONFUSIÓN Y ACCURACY
    # ════════════════════════════════════════════════════════════════════════

    # [D] Ajustamos el clf del CV sobre todo el set para poder predecir
    clf_cv.fit(X, y)
    y_pred = clf_cv.predict(X)   # [D] predicciones sobre el set completo

    # [D] ★ ACCURACY GLOBAL ★
    acc = accuracy_score(y, y_pred)
    print(f"\n  [D] ACCURACY GLOBAL (train set): {acc:.4f}  "
          f"← {acc*100:.1f}% de predicciones correctas")

    # [D] ★ TABLA DE CONFUSIÓN (resumen numérico) ★
    # Para 101 clases mostramos solo las métricas agregadas de la matriz
    cm = confusion_matrix(y, y_pred, labels=np.unique(y))
    vp_total = np.diag(cm).sum()           # suma de la diagonal = aciertos
    fp_total = cm.sum() - vp_total         # todo lo que no es diagonal = errores
    print(f"\n  [D] TABLA DE CONFUSIÓN (resumen):")
    print(f"      Verdaderos Positivos totales : {vp_total}")
    print(f"      Clasificaciones incorrectas  : {fp_total}")
    print(f"      Total de muestras            : {len(y)}")
    print(f"\n      Nota: la matriz completa es {cm.shape[0]}×{cm.shape[1]} "
          f"(una fila/columna por destino).")
    print(f"      El reporte por clase detalla precision/recall/f1 de cada celda:\n")

    # [D] ★ REPORTE DE CLASIFICACIÓN ★
    # Equivale a leer la tabla de confusión clase por clase:
    #   precision = VP_clase / (VP_clase + FP_clase)
    #   recall    = VP_clase / (VP_clase + FN_clase)
    print("  [D] REPORTE POR CLASE (Tabla de Confusión expandida):")
    print(classification_report(y, y_pred, zero_division=0))

    # [C] ★ ENTRENAMIENTO FINAL DEL MODELO ★
    clf_final = entrenar_final(X, y, vars_cfg)
    print("  ✔  Modelo final entrenado  ← [C] clf.fit() completado")

    # Guardar el modelo entrenado como .joblib
    guardar_modelo(
        clf_final, mlb_clima, mlb_gustos, scaler,
        vars_cfg,
        df[["nombre", "descripcion", "min_budget", "min_dias", "gustos", "clima"]]
    )

    print("\n🎉  ¡Listo! Ejecuta:  python model/recomendar.py\n")
