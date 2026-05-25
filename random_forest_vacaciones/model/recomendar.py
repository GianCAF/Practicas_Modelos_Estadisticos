"""
recomendar.py  —  Carga modelo_rf.joblib y recomienda destinos de vacaciones
=============================================================================
Uso:
    python model/recomendar.py

El script guía al usuario por 3 preguntas (clima, presupuesto+días, gustos),
consulta el modelo y muestra las mejores recomendaciones con descripción.
"""

import os
import sys
import yaml
import numpy as np
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_VARS   = os.path.join(BASE_DIR, "config", "variables.yml")
CFG_LIMS   = os.path.join(BASE_DIR, "config", "limites.yml")
MODEL_PATH = os.path.join(BASE_DIR, "model",  "modelo_rf.joblib")


# ── Helpers de consola ────────────────────────────────────────────────────────
def titulo(texto: str):
    print("\n" + "═" * 55)
    print(f"  {texto}")
    print("═" * 55)

def separador():
    print("─" * 55)

def error(msg: str):
    print(f"\n  ⚠️  {msg}\n")


# ── Cargar configuración ──────────────────────────────────────────────────────
def cargar_config():
    with open(CFG_VARS, "r", encoding="utf-8") as f:
        vars_cfg = yaml.safe_load(f)
    with open(CFG_LIMS, "r", encoding="utf-8") as f:
        lims_cfg = yaml.safe_load(f)
    return vars_cfg, lims_cfg


# ── Cargar modelo ─────────────────────────────────────────────────────────────
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("\n  ❌  No se encontró modelo_rf.joblib")
        print("      Ejecuta primero:  python model/entrenar.py\n")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
#  PREGUNTAS AL USUARIO
# ══════════════════════════════════════════════════════════════════════════════

def preguntar_clima(lims_cfg: dict) -> list[str]:
    """Sub-bosque 1: Clima preferido"""
    opciones = lims_cfg["clima"]["opciones"]

    print("\n  🌡️   ¿Qué tipo de clima prefieres para tus vacaciones?")
    separador()
    for op in opciones:
        print(f"    [{op['id']}]  {op['etiqueta']}")
    print(f"\n    Puedes elegir hasta {lims_cfg['clima']['maximo_seleccion']} opciones")
    print("    Ejemplo: 1  o  1,2")
    separador()

    while True:
        entrada = input("  Tu elección: ").strip()
        ids = [x.strip() for x in entrada.split(",") if x.strip()]
        valores = []
        validos = {str(op["id"]): op["valor"] for op in opciones}

        for id_str in ids:
            if id_str not in validos:
                error(f"Opción '{id_str}' no válida. Elige entre 1 y {len(opciones)}")
                valores = []
                break
            valores.append(validos[id_str])

        if not valores:
            continue
        if len(valores) > lims_cfg["clima"]["maximo_seleccion"]:
            error(lims_cfg["clima"]["mensaje_invalido"])
            continue
        return valores


def preguntar_presupuesto_dias(lims_cfg: dict) -> tuple[int, int]:
    """Sub-bosque 2: Presupuesto y días"""
    p_min = lims_cfg["presupuesto"]["minimo"]
    p_max = lims_cfg["presupuesto"]["maximo"]
    d_min = lims_cfg["dias"]["minimo"]
    d_max = lims_cfg["dias"]["maximo"]

    print(f"\n  💰  ¿Cuál es tu presupuesto aproximado? (MXN)")
    print(f"      Rango: ${p_min:,} – ${p_max:,}")
    separador()

    while True:
        entrada = input("  Presupuesto: $").strip().replace(",", "")
        try:
            budget = int(float(entrada))
            if budget < p_min or budget > p_max:
                error(lims_cfg["presupuesto"]["mensaje_fuera_rango"])
                continue
            break
        except ValueError:
            error("Ingresa un número válido (ej. 5000)")

    print(f"\n  📅  ¿Cuántos días tienes disponibles?")
    print(f"      Rango: {d_min} – {d_max} días")
    separador()

    while True:
        entrada = input("  Días: ").strip()
        try:
            dias = int(entrada)
            if dias < d_min or dias > d_max:
                error(lims_cfg["dias"]["mensaje_fuera_rango"])
                continue
            break
        except ValueError:
            error("Ingresa un número entero (ej. 3)")

    return budget, dias


def preguntar_gustos(lims_cfg: dict) -> list[str]:
    """Sub-bosque 3: Gustos personales"""
    opciones = lims_cfg["gustos"]["opciones"]
    g_min    = lims_cfg["gustos"]["minimo_seleccion"]
    g_max    = lims_cfg["gustos"]["maximo_seleccion"]

    print(f"\n  🎯  ¿Qué actividades o experiencias te interesan?")
    separador()
    for op in opciones:
        print(f"    [{op['id']}]  {op['etiqueta']}")
    print(f"\n    Elige entre {g_min} y {g_max} opciones")
    print("    Ejemplo: 1,3,5")
    separador()

    while True:
        entrada = input("  Tu elección: ").strip()
        ids = [x.strip() for x in entrada.split(",") if x.strip()]
        valores = []
        validos = {str(op["id"]): op["valor"] for op in opciones}

        for id_str in ids:
            if id_str not in validos:
                error(f"Opción '{id_str}' no válida.")
                valores = []
                break
            valores.append(validos[id_str])

        if not valores:
            continue
        if not (g_min <= len(valores) <= g_max):
            error(lims_cfg["gustos"]["mensaje_invalido"])
            continue
        return valores


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL VECTOR DE CONSULTA
# ══════════════════════════════════════════════════════════════════════════════

def construir_vector_consulta(climas: list[str], budget: int, dias: int,
                               gustos: list[str], artefacto: dict) -> np.ndarray:
    vars_cfg   = artefacto["vars_cfg"]
    mlb_clima  = artefacto["mlb_clima"]
    mlb_gustos = artefacto["mlb_gustos"]
    scaler     = artefacto["scaler"]

    # Sub-bosque clima
    X_clima = mlb_clima.transform([climas])

    # Sub-bosque presupuesto + días
    X_pres = scaler.transform([[budget, dias]])

    # Sub-bosque gustos
    X_gustos = mlb_gustos.transform([gustos])

    # Pesos
    w_c = vars_cfg["sub_bosques"]["peso_clima"]
    w_p = vars_cfg["sub_bosques"]["peso_presupuesto_tiempo"]
    w_g = vars_cfg["sub_bosques"]["peso_gustos"]

    return np.hstack([X_clima * w_c, X_pres * w_p, X_gustos * w_g])


# ══════════════════════════════════════════════════════════════════════════════
#  RECOMENDACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def recomendar(X_query: np.ndarray, artefacto: dict,
               lims_cfg: dict, budget: int) -> list[dict]:
    clf       = artefacto["modelo"]
    df_dest   = artefacto["df_destinos"]
    vars_cfg  = artefacto["vars_cfg"]
    top_n     = vars_cfg["output"]["top_n_recomendaciones"]
    confianza = lims_cfg["modelo"]["confianza_minima"]
    adv_pres  = lims_cfg["modelo"]["advertencia_presupuesto"]

    proba      = clf.predict_proba(X_query)[0]
    clases     = clf.classes_
    pares      = sorted(zip(clases, proba), key=lambda x: x[1], reverse=True)

    resultados = []
    for nombre, score in pares[:top_n * 2]:   # candidatos extra por si filtran
        if score < confianza:
            continue
        fila = df_dest[df_dest["nombre"] == nombre]
        if fila.empty:
            continue
        fila = fila.iloc[0]

        pct_budget = fila["min_budget"] / budget
        advertencia = None
        if pct_budget > adv_pres:
            advertencia = (
                f"⚠️  Este destino usa el {pct_budget*100:.0f}% de tu presupuesto"
            )

        resultados.append({
            "nombre"      : nombre,
            "score"       : score,
            "descripcion" : fila["descripcion"],
            "min_budget"  : int(fila["min_budget"]),
            "min_dias"    : int(fila["min_dias"]),
            "clima"       : fila["clima"],
            "gustos"      : fila["gustos"],
            "advertencia" : advertencia,
        })

        if len(resultados) >= top_n:
            break

    # Fallback si no hay suficientes
    if not resultados:
        fallback_n = lims_cfg["modelo"]["fallback_top_n"]
        top_idx    = np.argsort(proba)[::-1][:fallback_n]
        for i in top_idx:
            fila = df_dest[df_dest["nombre"] == clases[i]]
            if fila.empty:
                continue
            fila = fila.iloc[0]
            resultados.append({
                "nombre"      : clases[i],
                "score"       : proba[i],
                "descripcion" : fila["descripcion"],
                "min_budget"  : int(fila["min_budget"]),
                "min_dias"    : int(fila["min_dias"]),
                "clima"       : fila["clima"],
                "gustos"      : fila["gustos"],
                "advertencia" : "ℹ️  Recomendación general (baja coincidencia con tu perfil)",
            })

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
#  MOSTRAR RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def mostrar_resultados(resultados: list[dict]):
    titulo("🌴  TUS DESTINOS RECOMENDADOS")

    for i, r in enumerate(resultados, 1):
        print(f"\n  #{i}  📍 {r['nombre']}")
        print(f"       Coincidencia : {r['score']*100:.1f}%")
        print(f"       Presupuesto  : ${r['min_budget']:,} MXN mínimo")
        print(f"       Días         : {r['min_dias']} días mínimo")
        print(f"       Clima        : {r['clima']}")
        print(f"       Gustos       : {r['gustos']}")
        print(f"\n       {r['descripcion']}")
        if r["advertencia"]:
            print(f"\n       {r['advertencia']}")
        separador()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    titulo("🧳  RECOMENDADOR DE VACACIONES EN MÉXICO")
    print("  Basado en Random Forest — 3 sub-bosques de decisión")
    print("  [Clima | Presupuesto+Tiempo | Gustos]")

    vars_cfg, lims_cfg = cargar_config()
    artefacto          = cargar_modelo()

    print("\n  ✅  Modelo listo. Responde las siguientes preguntas:\n")

    # ── Preguntas por sub-bosque ──────────────────────────────────────────────
    titulo("🌡️   SUB-BOSQUE 1 — CLIMA")
    climas = preguntar_clima(lims_cfg)

    titulo("💰  SUB-BOSQUE 2 — PRESUPUESTO Y TIEMPO")
    budget, dias = preguntar_presupuesto_dias(lims_cfg)

    titulo("🎯  SUB-BOSQUE 3 — GUSTOS")
    gustos = preguntar_gustos(lims_cfg)

    # ── Inferencia ────────────────────────────────────────────────────────────
    print("\n  🤖  Consultando el bosque...")
    X_q = construir_vector_consulta(climas, budget, dias, gustos, artefacto)
    resultados = recomendar(X_q, artefacto, lims_cfg, budget)

    # ── Resultados ────────────────────────────────────────────────────────────
    mostrar_resultados(resultados)

    print("\n  ¿Deseas buscar otro destino? [s/n]: ", end="")
    if input().strip().lower() == "s":
        main()
    else:
        print("\n  ¡Buen viaje! 🌴✈️\n")


if __name__ == "__main__":
    main()
