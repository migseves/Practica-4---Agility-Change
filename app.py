# =============================================================================
# Pronóstico de entregas JMS — Versión 5
# App de Streamlit en un solo archivo. Ejecutar con:  streamlit run app.py
# v1: pestaña "Cargar" (CSV) + validación de protección + conteo por estado.
# v2: pestaña "Historial" (tareas terminadas por semana, promedio, tendencias,
#     filtros, fecha de corte y semanas configurables, aviso de excluidas).
# v3: pestaña "Pronóstico" (Monte Carlo 10.000 simulaciones, tarjetas 50/85/95 %,
#     filtro por tipo, gráficos de apoyo y comparación entre tipos).
# v4: en la pestaña "Pronóstico", sección «¿Cuántas tareas alcanzamos antes de…?»
#     (fecha límite fija → al menos X tareas al 50 / 85 / 95 %, gráfico de las
#     10.000 sumas).
# v5: pestaña "Tiempo de ciclo" («Si empezamos esta tarea hoy, ¿cuándo estará
#     lista?»): promedio, 50 % y 85 %, frase resumen y gráfico de días.
# =============================================================================

import io
import math
import re
import unicodedata
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Compatibilidad: Streamlit reciente usa width="stretch"; versiones antiguas, use_container_width.
_VER = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
ANCHO = {"width": "stretch"} if _VER >= (1, 50) else {"use_container_width": True}

# -----------------------------------------------------------------------------
# 0. CONFIGURACIÓN
# -----------------------------------------------------------------------------
APP_TITULO = "Pronóstico de entregas JMS"
APP_VERSION = "Versión 5"

COLOR_AZUL = "#0F265C"       # Azul Banco Pichincha
COLOR_AMARILLO = "#FFDD00"   # Amarillo Banco Pichincha
COLOR_GRIS = "#5A6673"
COLOR_GRIS_CLARO = "#C9CFD9"
COLOR_AZUL_CLARO = "#4A6FB5"
COLOR_FONDO = "#F4F6FA"

TAMANO_MAX_MB = 10

COLUMNAS_REQUERIDAS = [
    "clave", "resumen", "tipo", "estado", "responsable",
    "informador", "creada", "fecha_inicio", "fecha_fin", "comentario",
]

MENSAJE_NO_PROTEGIDO = "Este archivo no está protegido. Pásalo primero por tu Protector"

# Patrones que delatan datos sensibles sin proteger
PATRONES_SENSIBLES = [
    ("@", re.compile(r"@")),
    ("cel. seguido de número", re.compile(r"\bcel\.\s*\d", re.IGNORECASE)),
    ("CC seguido de número", re.compile(r"\bCC\.?\s*:?\s*\d", re.IGNORECASE)),
]

# Sinónimos aceptados (se comparan sin tildes, en minúsculas, espacios simples)
SINONIMOS_ESTADO = {
    "Finalizada": ["finalizada", "finalizado", "terminada", "terminado", "hecho",
                   "hecha", "done", "cerrada", "cerrado", "completada", "completado"],
    "En curso": ["en curso", "en progreso", "in progress", "doing", "en proceso"],
    "Por hacer": ["por hacer", "pendiente", "to do", "todo", "backlog", "abierta", "abierto"],
}
ORDEN_ESTADOS = ["Finalizada", "En curso", "Por hacer"]

SINONIMOS_TIPO = {
    "Historia": ["historia", "historia de usuario", "hu", "story", "user story"],
    "Error": ["error", "bug", "defecto", "incidente", "incidencia"],
    "Tarea": ["tarea", "task", "subtarea", "sub-tarea", "subtask"],
}
ORDEN_TIPOS = ["Historia", "Error", "Tarea"]

# Historial (receta)
FECHA_CORTE_DEFECTO = date(2026, 8, 31)   # lunes 31 de agosto de 2026 ("hoy")
SEMANAS_DEFECTO = 26
SEMANAS_MIN, SEMANAS_MAX = 4, 52
VENTANA_MEDIA_MOVIL = 4

# Pronóstico (receta)
FECHA_COMITE_DEFECTO = date(2026, 11, 30)
SIMULACIONES = 10_000
SEMILLA = 42                     # fija: el resultado se puede reproducir en el comité
SEMANAS_TOPE = 520               # 10 años: si no se alcanza, se marca "sin fecha"
PERCENTILES = [
    # (probabilidad, etiqueta, subtítulo, color fondo, color texto, color acento)
    (0.50, "50 %", "Probable", "#FFFFFF", COLOR_AZUL, COLOR_GRIS),
    (0.85, "85 %", "Recomendado", COLOR_AMARILLO, COLOR_AZUL, COLOR_AZUL),
    (0.95, "95 %", "El más seguro", COLOR_AZUL, "#FFFFFF", COLOR_AMARILLO),
]

# «¿Cuántas alcanzamos?» (v4)
FECHA_LIMITE_DEFECTO = date(2026, 11, 29)   # domingo: 13 semanas completas desde el corte
ESCENARIOS_ALCANCE = [
    # (probabilidad, etiqueta, subtítulo, fondo, texto, acento)
    (0.50, "50 %", "Probable", "#FFFFFF", COLOR_AZUL, COLOR_GRIS),
    (0.85, "85 %", "Recomendado", COLOR_AMARILLO, COLOR_AZUL, COLOR_AZUL),
    (0.95, "95 %", "El más seguro", COLOR_AZUL, "#FFFFFF", COLOR_AMARILLO),
]

MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def fmt_fecha(d) -> str:
    """02-mar-2026"""
    d = pd.Timestamp(d)
    return f"{d.day:02d}-{MESES[d.month - 1]}-{d.year}"


def fmt_fecha_larga(d) -> str:
    """lunes 02-mar-2026"""
    d = pd.Timestamp(d)
    return f"{DIAS[d.weekday()]} {fmt_fecha(d)}"


def fmt_num(x, decimales=1) -> str:
    """Número con coma decimal y punto de miles (formato es-EC)."""
    texto = f"{x:,.{decimales}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


# -----------------------------------------------------------------------------
# 1. CARGA DEL ARCHIVO
# -----------------------------------------------------------------------------
def leer_csv(contenido: bytes) -> pd.DataFrame:
    """Lee un CSV detectando separador (',' o ';') y codificación."""
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            texto = contenido.decode(codificacion)
            df = pd.read_csv(
                io.StringIO(texto), sep=None, engine="python",
                dtype=str, keep_default_na=False,
            )
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo leer el archivo: codificación no reconocida.")


def validar_estructura(df: pd.DataFrame) -> list:
    """Devuelve la lista de errores de estructura (vacía si todo está bien)."""
    errores = []
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        errores.append("Faltan columnas obligatorias: " + ", ".join(faltantes))
    if df.empty:
        errores.append("El archivo no tiene filas de datos.")
    return errores


# -----------------------------------------------------------------------------
# 2. PROTECCIÓN
# -----------------------------------------------------------------------------
def buscar_datos_sensibles(df: pd.DataFrame) -> list:
    """Revisa TODAS las celdas. Devuelve hallazgos (fila, columna, motivo)."""
    hallazgos = []
    for columna in df.columns:
        for indice, valor in df[columna].items():
            texto = str(valor)
            for motivo, patron in PATRONES_SENSIBLES:
                if patron.search(texto):
                    # +2: una por el encabezado y otra porque la hoja cuenta desde 1
                    hallazgos.append({"fila": indice + 2, "columna": columna, "motivo": motivo})
                    break
    return hallazgos


# -----------------------------------------------------------------------------
# 3. NORMALIZACIÓN
# -----------------------------------------------------------------------------
def _clave_texto(valor) -> str:
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(texto.lower().split())


def _normalizar(valor, sinonimos: dict) -> str:
    clave = _clave_texto(valor)
    for canonico, lista in sinonimos.items():
        if clave in lista:
            return canonico
    return str(valor).strip() or "(vacío)"


def normalizar_estado(valor) -> str:
    return _normalizar(valor, SINONIMOS_ESTADO)


def normalizar_tipo(valor) -> str:
    return _normalizar(valor, SINONIMOS_TIPO)


def parsear_fecha(serie: pd.Series) -> pd.Series:
    """Fechas en formato AAAA-MM-DD (acepta hora al final). Inválidas → NaT."""
    return pd.to_datetime(
        serie.astype(str).str.strip().str[:10], format="%Y-%m-%d", errors="coerce"
    )


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas normalizadas sin tocar las originales."""
    datos = df.copy()
    datos["estado_n"] = datos["estado"].map(normalizar_estado)
    datos["tipo_n"] = datos["tipo"].map(normalizar_tipo)
    datos["responsable_n"] = datos["responsable"].astype(str).str.strip().replace("", "(sin asignar)")
    for col in ("creada", "fecha_inicio", "fecha_fin"):
        datos[f"{col}_dt"] = parsear_fecha(datos[col])
    return datos


def tipos_presentes(datos: pd.DataFrame) -> list:
    presentes = list(datos["tipo_n"].unique())
    conocidos = [t for t in ORDEN_TIPOS if t in presentes]
    otros = sorted(t for t in presentes if t not in ORDEN_TIPOS)
    return conocidos + otros


# -----------------------------------------------------------------------------
# 4. MÉTRICAS — HISTORIAL
# -----------------------------------------------------------------------------
def contar_por_estado(datos: pd.DataFrame) -> pd.DataFrame:
    conteo = datos["estado_n"].value_counts().rename_axis("Estado").reset_index(name="Tareas")
    orden = {e: i for i, e in enumerate(ORDEN_ESTADOS)}
    conteo["_orden"] = conteo["Estado"].map(lambda e: orden.get(e, 99))
    return conteo.sort_values(["_orden", "Estado"]).drop(columns="_orden").reset_index(drop=True)


def lunes_de(fecha) -> pd.Timestamp:
    f = pd.Timestamp(fecha)
    return f - pd.Timedelta(days=f.weekday())


def ventana_historial(fecha_corte: date, semanas: int):
    """
    Semanas completas (lunes a domingo) anteriores a la fecha de corte.
    Devuelve (inicio_lunes, fin_domingo) como Timestamps.
    """
    lunes_semana_corte = lunes_de(fecha_corte)
    fin = lunes_semana_corte - pd.Timedelta(days=1)            # domingo anterior
    inicio = lunes_semana_corte - pd.Timedelta(weeks=semanas)  # lunes N semanas atrás
    return inicio, fin


def calcular_historial(datos: pd.DataFrame, inicio: pd.Timestamp, fin: pd.Timestamp):
    """
    Receta:
      - Cuenta tareas Finalizadas cuya fecha_fin cae en cada semana (lunes-domingo).
      - Semanas sin tareas cuentan como 0.
      - Excluye: finalizadas sin fecha_fin (o fecha no válida), terminadas antes
        del historial y terminadas después del fin del historial.
    Devuelve (tabla_semanal, excluidas_df).
    """
    finalizadas = datos[datos["estado_n"] == "Finalizada"].copy()
    ff = finalizadas["fecha_fin_dt"]

    motivo = pd.Series("", index=finalizadas.index)
    motivo[ff.isna()] = "Finalizada sin fecha_fin (o con fecha no válida)"
    motivo[ff.notna() & (ff < inicio)] = f"Terminada antes del historial (antes del {fmt_fecha(inicio)})"
    motivo[ff.notna() & (ff > fin)] = (
        f"Terminada después del historial (desde el {fmt_fecha(fin + pd.Timedelta(days=1))})"
    )

    excluidas = finalizadas[motivo != ""].assign(motivo=motivo[motivo != ""])
    validas = finalizadas[motivo == ""]

    semanas = pd.date_range(inicio, fin, freq="W-MON")
    lunes_de_cada_tarea = validas["fecha_fin_dt"] - pd.to_timedelta(
        validas["fecha_fin_dt"].dt.weekday, unit="D"
    )
    conteo = lunes_de_cada_tarea.value_counts().reindex(semanas, fill_value=0)

    tabla = pd.DataFrame({
        "inicio_semana": semanas,
        "fin_semana": semanas + pd.Timedelta(days=6),
        "terminadas": conteo.values.astype(int),
    })
    return tabla, excluidas


def estadisticas(tabla: pd.DataFrame) -> dict:
    valores = tabla["terminadas"]
    promedio = float(valores.mean()) if len(valores) else 0.0
    desv = float(valores.std(ddof=0)) if len(valores) else 0.0
    return {
        "semanas": len(valores),
        "total": int(valores.sum()),
        "promedio": promedio,
        "mediana": float(valores.median()) if len(valores) else 0.0,
        "minimo": int(valores.min()) if len(valores) else 0,
        "maximo": int(valores.max()) if len(valores) else 0,
        "semanas_cero": int((valores == 0).sum()),
        "variabilidad": (desv / promedio) if promedio > 0 else 0.0,
    }


# -----------------------------------------------------------------------------
# 5. MONTE CARLO — PRONÓSTICO
# -----------------------------------------------------------------------------
def contar_pendientes(datos: pd.DataFrame) -> pd.DataFrame:
    """Pendientes = todas las tareas cuyo estado NO es Finalizada."""
    return datos[datos["estado_n"] != "Finalizada"]


@st.cache_data(show_spinner=False)
def simular_semanas(throughput: tuple, pendientes: int,
                    simulaciones: int = SIMULACIONES, semilla: int = SEMILLA) -> np.ndarray:
    """
    Receta Monte Carlo:
      Repetir `simulaciones` veces: sacar al azar una semana del historial
      (con reemplazo) y sumar sus tareas hasta alcanzar las pendientes.
      Anotar cuántas semanas se necesitaron.
    Devuelve un arreglo con las semanas necesarias en cada simulación.
    Si una simulación no alcanza las pendientes en SEMANAS_TOPE, se anota SEMANAS_TOPE + 1.
    """
    muestras = np.asarray(throughput, dtype=np.int64)
    if pendientes <= 0:
        return np.zeros(simulaciones, dtype=np.int64)
    if len(muestras) == 0 or muestras.sum() == 0:
        return np.full(simulaciones, SEMANAS_TOPE + 1, dtype=np.int64)

    rng = np.random.default_rng(semilla)
    resultado = np.full(simulaciones, SEMANAS_TOPE + 1, dtype=np.int64)
    acumulado = np.zeros(simulaciones, dtype=np.int64)
    activas = np.arange(simulaciones)

    # Se avanza en bloques de semanas para no reservar memoria innecesaria
    semana_actual = 0
    bloque = 26
    while len(activas) and semana_actual < SEMANAS_TOPE:
        n_bloque = min(bloque, SEMANAS_TOPE - semana_actual)
        sorteo = rng.choice(muestras, size=(len(activas), n_bloque), replace=True)
        suma = acumulado[activas][:, None] + np.cumsum(sorteo, axis=1)
        alcanzo = suma >= pendientes
        tiene = alcanzo.any(axis=1)
        primera = alcanzo.argmax(axis=1)
        resultado[activas[tiene]] = semana_actual + primera[tiene] + 1
        acumulado[activas] = suma[:, -1]
        activas = activas[~tiene]
        semana_actual += n_bloque
    return resultado


def valor_en_posicion(resultados: np.ndarray, probabilidad: float) -> int:
    """Ordena los resultados y toma el que queda en la posición p (p. ej. 8.500 de 10.000)."""
    ordenados = np.sort(resultados)
    posicion = max(1, math.ceil(probabilidad * len(ordenados)))
    return int(ordenados[posicion - 1])


def fecha_fin_semana_n(fecha_corte, n: int) -> pd.Timestamp:
    """Domingo de la semana N contando desde la fecha de corte (N = 1 → domingo de esa semana)."""
    return lunes_de(fecha_corte) + pd.Timedelta(days=7 * n - 1)


def pronosticar(datos: pd.DataFrame, inicio, fin, fecha_corte) -> dict:
    """Ejecuta el pronóstico completo para un subconjunto de datos."""
    pendientes_df = contar_pendientes(datos)
    pendientes = len(pendientes_df)
    tabla, _ = calcular_historial(datos, inicio, fin)
    throughput = tuple(int(x) for x in tabla["terminadas"])
    resultados = simular_semanas(throughput, pendientes)

    percentiles = []
    for p, *_ in PERCENTILES:
        n = valor_en_posicion(resultados, p)
        sin_fecha = n > SEMANAS_TOPE
        percentiles.append({
            "p": p,
            "semanas": n,
            "sin_fecha": sin_fecha,
            "fecha": None if (sin_fecha or pendientes == 0) else fecha_fin_semana_n(fecha_corte, n),
        })
    return {
        "pendientes": pendientes,
        "en_curso": int((pendientes_df["estado_n"] == "En curso").sum()),
        "por_hacer": int((pendientes_df["estado_n"] == "Por hacer").sum()),
        "otros": int((~pendientes_df["estado_n"].isin(["En curso", "Por hacer"])).sum()),
        "tabla": tabla,
        "throughput": throughput,
        "promedio": float(np.mean(throughput)) if throughput else 0.0,
        "semanas_historial": len(throughput),
        "resultados": resultados,
        "percentiles": percentiles,
    }


def probabilidad_antes_de(resultados: np.ndarray, fecha_corte, fecha_objetivo) -> float:
    """% de simulaciones que terminan el domingo de la semana N <= fecha objetivo."""
    if len(resultados) == 0:
        return 0.0
    semanas_disponibles = (pd.Timestamp(fecha_objetivo) - lunes_de(fecha_corte)).days // 7
    # Si la fecha objetivo cae a mitad de semana, solo cuentan las semanas completas.
    domingo_n = fecha_fin_semana_n(fecha_corte, semanas_disponibles + 1)
    if domingo_n <= pd.Timestamp(fecha_objetivo):
        semanas_disponibles += 1
    return float((resultados <= semanas_disponibles).mean())


# --- «¿Cuántas tareas alcanzamos antes de…?» (v4) ----------------------------
def semanas_disponibles(fecha_corte, fecha_limite) -> int:
    """
    Semanas completas (lunes a domingo) entre la fecha de corte y la fecha límite.
    Semana 1 = la que empieza el lunes de la fecha de corte.
    Ej.: corte 31-ago-2026 y límite 29-nov-2026 → 13 semanas.
    Si la fecha límite cae a mitad de semana, esa semana no cuenta.
    """
    dias = (pd.Timestamp(fecha_limite) - lunes_de(fecha_corte)).days + 1
    return max(0, dias // 7)


@st.cache_data(show_spinner=False)
def simular_sumas(throughput: tuple, semanas: int,
                  simulaciones: int = SIMULACIONES, semilla: int = SEMILLA) -> np.ndarray:
    """
    Receta: repetir `simulaciones` veces → sacar al azar `semanas` semanas del
    historial (pueden repetirse) y sumar sus tareas. Devuelve las 10.000 sumas.
    """
    muestras = np.asarray(throughput, dtype=np.int64)
    if semanas <= 0 or len(muestras) == 0:
        return np.zeros(simulaciones, dtype=np.int64)
    rng = np.random.default_rng(semilla)
    return rng.choice(muestras, size=(simulaciones, semanas), replace=True).sum(axis=1)


def posicion_al_menos(probabilidad: float, total: int = SIMULACIONES) -> int:
    """85 % → posición 15 % (n.º 1.500); 95 % → 5 % (n.º 500); 50 % → la mitad (n.º 5.000)."""
    return max(1, int(round((1 - probabilidad) * total)))


def al_menos(sumas: np.ndarray, probabilidad: float) -> int:
    """Ordena las sumas de menor a mayor y toma la de la posición (1 − p)."""
    ordenadas = np.sort(sumas)
    return int(ordenadas[posicion_al_menos(probabilidad, len(ordenadas)) - 1])


def pronosticar_alcance(datos: pd.DataFrame, inicio, fin, fecha_corte, fecha_limite) -> dict:
    tabla, _ = calcular_historial(datos, inicio, fin)
    throughput = tuple(int(x) for x in tabla["terminadas"])
    n = semanas_disponibles(fecha_corte, fecha_limite)
    sumas = simular_sumas(throughput, n)
    return {
        "semanas": n,
        "ultimo_domingo": fecha_fin_semana_n(fecha_corte, n) if n else None,
        "throughput": throughput,
        "semanas_historial": len(throughput),
        "promedio": float(np.mean(throughput)) if throughput else 0.0,
        "pendientes": len(contar_pendientes(datos)),
        "sumas": sumas,
        "escenarios": [
            {"p": p, "cantidad": al_menos(sumas, p), "posicion": posicion_al_menos(p)}
            for p, *_ in ESCENARIOS_ALCANCE
        ],
    }


# -----------------------------------------------------------------------------
# 5b. TIEMPO DE CICLO (v5)
# -----------------------------------------------------------------------------
def calcular_tiempo_ciclo(datos: pd.DataFrame, inicio, fin):
    """
    Receta:
      Tiempo de ciclo = días de calendario entre fecha_inicio y fecha_fin
      (solo las fechas, sin las horas), para las tareas válidas del historial.
    Tareas válidas = Finalizadas cuya fecha_fin cae dentro del historial
    y que además tienen una fecha_inicio válida y no posterior a fecha_fin.
    Devuelve (validas_df con columna 'dias', excluidas_df con columna 'motivo').
    """
    finalizadas = datos[datos["estado_n"] == "Finalizada"].copy()
    fi = finalizadas["fecha_inicio_dt"].dt.normalize()
    ff = finalizadas["fecha_fin_dt"].dt.normalize()

    motivo = pd.Series("", index=finalizadas.index)
    motivo[ff.isna()] = "Finalizada sin fecha_fin (o con fecha no válida)"
    motivo[(motivo == "") & (ff < inicio)] = f"Terminada antes del historial (antes del {fmt_fecha(inicio)})"
    motivo[(motivo == "") & (ff > fin)] = (
        f"Terminada después del historial (desde el {fmt_fecha(fin + pd.Timedelta(days=1))})"
    )
    motivo[(motivo == "") & fi.isna()] = "Sin fecha_inicio (o con fecha no válida)"
    motivo[(motivo == "") & (ff < fi)] = "fecha_fin anterior a fecha_inicio"

    excluidas = finalizadas[motivo != ""].assign(motivo=motivo[motivo != ""])
    validas = finalizadas[motivo == ""].copy()
    validas["dias"] = (ff[motivo == ""] - fi[motivo == ""]).dt.days.astype(int)
    return validas, excluidas


def valor_bajo_el_cual(valores, probabilidad: float):
    """Valor bajo el cual (o igual) queda el p % de las tareas: se ordenan y se toma la posición p."""
    ordenados = np.sort(np.asarray(valores))
    if len(ordenados) == 0:
        return None
    posicion = max(1, math.ceil(probabilidad * len(ordenados)))
    return int(ordenados[posicion - 1])


def resumen_ciclo(dias) -> dict:
    dias = np.asarray(dias)
    if len(dias) == 0:
        return {"n": 0, "promedio": None, "p50": None, "p85": None, "minimo": None, "maximo": None}
    return {
        "n": int(len(dias)),
        "promedio": float(dias.mean()),
        "p50": valor_bajo_el_cual(dias, 0.50),
        "p85": valor_bajo_el_cual(dias, 0.85),
        "minimo": int(dias.min()),
        "maximo": int(dias.max()),
    }


# -----------------------------------------------------------------------------
# 6. INTERFAZ — ESTILO Y PARÁMETROS
# -----------------------------------------------------------------------------
def aplicar_estilo():
    st.markdown(
        f"""
        <style>
        h1, h2, h3, h4 {{ color: {COLOR_AZUL}; }}
        .stButton > button {{
            background-color: {COLOR_AMARILLO}; color: {COLOR_AZUL};
            font-weight: 700; border: none;
        }}
        .stButton > button:hover {{ background-color: {COLOR_AZUL}; color: white; }}
        .caja-requisitos {{
            border-left: 6px solid {COLOR_AMARILLO}; background: {COLOR_FONDO};
            padding: 0.8rem 1rem; border-radius: 6px; color: {COLOR_AZUL};
        }}
        .tarjeta {{
            border-radius: 14px; padding: 1.1rem 1.2rem 0.9rem 1.2rem;
            border: 2px solid {COLOR_AZUL}; height: 100%;
            box-shadow: 0 2px 6px rgba(15,38,92,0.12);
        }}
        .tarjeta .prob {{ font-size: 2.6rem; font-weight: 800; line-height: 1; }}
        .tarjeta .sub {{ font-size: 0.95rem; font-weight: 700; letter-spacing: .04em;
                         text-transform: uppercase; margin-top: .2rem; }}
        .tarjeta .fecha {{ font-size: 1.55rem; font-weight: 800; margin-top: .8rem; }}
        .tarjeta .dia {{ font-size: 0.9rem; opacity: .85; }}
        .tarjeta .semanas {{ font-size: 1rem; margin-top: .3rem; }}
        .tarjeta .origen {{ font-size: 0.8rem; margin-top: .8rem; padding-top: .6rem;
                            border-top: 1px solid rgba(128,128,128,.35); opacity: .9; }}
        .insignia {{ display:inline-block; padding: 2px 8px; border-radius: 10px;
                     font-size: .72rem; font-weight: 800; margin-left: 6px;
                     vertical-align: middle; }}
        .barra-prob {{ height: 8px; border-radius: 4px; background: rgba(128,128,128,.25);
                       margin-top: .6rem; overflow: hidden; }}
        .barra-prob > div {{ height: 100%; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def panel_parametros():
    """Barra lateral: fecha de corte, semanas de historial y fecha del comité."""
    with st.sidebar:
        st.header("Parámetros")
        fecha_corte = st.date_input(
            "Fecha de corte («hoy»)", value=FECHA_CORTE_DEFECTO, format="YYYY-MM-DD",
            help="El historial usa las semanas completas anteriores a esta fecha.",
        )
        semanas = st.number_input(
            "Semanas de historial", min_value=SEMANAS_MIN, max_value=SEMANAS_MAX,
            value=SEMANAS_DEFECTO, step=1,
            help="Por defecto 26. El archivo de control usa 10.",
        )
        inicio, fin = ventana_historial(fecha_corte, int(semanas))
        st.caption(
            f"Historial: del **{fmt_fecha_larga(inicio)}** "
            f"al **{fmt_fecha_larga(fin)}** ({int(semanas)} semanas)."
        )
        st.divider()
        fecha_comite = st.date_input(
            "Fecha que pregunta el comité", value=FECHA_COMITE_DEFECTO, format="YYYY-MM-DD",
            help="Se usa en el Pronóstico para calcular la probabilidad de terminar a tiempo.",
        )
    return fecha_corte, int(semanas), inicio, fin, fecha_comite


# -----------------------------------------------------------------------------
# 7. INTERFAZ — CARGAR
# -----------------------------------------------------------------------------
def pestana_cargar():
    st.subheader("Cargar archivo")

    st.markdown(
        f"""
        <div class="caja-requisitos">
        <b>Requisitos del archivo</b><br>
        • Tipo: <b>CSV</b> (separado por coma o punto y coma)<br>
        • Número de archivos: <b>1</b><br>
        • Tamaño máximo: <b>{TAMANO_MAX_MB} MB</b><br>
        • Debe haber pasado por el <b>Protector</b><br>
        • Fechas en formato <b>AAAA-MM-DD</b><br>
        • Columnas: {", ".join(COLUMNAS_REQUERIDAS)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    archivo = st.file_uploader("Elige un archivo CSV", type=["csv"], accept_multiple_files=False)

    # Si cambia el archivo, se limpia el resultado anterior
    nombre_actual = archivo.name if archivo else None
    if st.session_state.get("nombre_archivo") != nombre_actual:
        st.session_state["nombre_archivo"] = nombre_actual
        st.session_state["datos"] = None

    if archivo is None:
        st.info("Selecciona un archivo y pulsa **Validar**.")
        return

    if st.button("Validar"):
        st.session_state["datos"] = None

        if archivo.size > TAMANO_MAX_MB * 1024 * 1024:
            st.error(f"❌ Archivo rechazado: supera el máximo de {TAMANO_MAX_MB} MB.")
            return

        try:
            df = leer_csv(archivo.getvalue())
        except Exception as error:
            st.error(f"❌ Archivo rechazado: no se pudo leer como CSV. Detalle: {error}")
            return

        # Protección (se revisa antes que todo lo demás)
        hallazgos = buscar_datos_sensibles(df)
        if hallazgos:
            st.error(f"❌ {MENSAJE_NO_PROTEGIDO}")
            with st.expander(f"Ver dónde se detectó ({len(hallazgos)} celdas)"):
                st.dataframe(pd.DataFrame(hallazgos), hide_index=True, **ANCHO)
            return

        errores = validar_estructura(df)
        if errores:
            st.error("❌ Archivo rechazado:\n\n" + "\n".join(f"- {e}" for e in errores))
            return

        st.session_state["datos"] = preparar_datos(df)

    datos = st.session_state.get("datos")
    if datos is None:
        return

    st.success(f"✅ Archivo aceptado: **{archivo.name}** · {len(datos)} tareas")

    conteo = contar_por_estado(datos)
    st.markdown("#### Tareas por estado")

    columnas = st.columns(len(conteo))
    for col, (_, fila) in zip(columnas, conteo.iterrows()):
        col.metric(fila["Estado"], int(fila["Tareas"]))

    izquierda, derecha = st.columns([1, 2])
    with izquierda:
        st.dataframe(conteo, hide_index=True, **ANCHO)
    with derecha:
        st.bar_chart(conteo.set_index("Estado")["Tareas"], color=COLOR_AZUL)

    no_reconocidos = [e for e in conteo["Estado"] if e not in ORDEN_ESTADOS]
    if no_reconocidos:
        st.warning("Estados no reconocidos: " + ", ".join(no_reconocidos))

    st.info("👉 Ahora abre las pestañas **Historial** y **Pronóstico**.")


# -----------------------------------------------------------------------------
# 8. INTERFAZ — HISTORIAL
# -----------------------------------------------------------------------------
def grafico_historial(tabla: pd.DataFrame, promedio: float) -> go.Figure:
    etiquetas = [fmt_fecha(d)[:6] for d in tabla["inicio_semana"]]
    textos_hover = [
        f"{fmt_fecha(a)} a {fmt_fecha(b)}" for a, b in zip(tabla["inicio_semana"], tabla["fin_semana"])
    ]
    y = tabla["terminadas"].to_numpy()

    fig = go.Figure()
    fig.add_bar(
        x=etiquetas, y=y, name="Terminadas", marker_color=COLOR_AZUL,
        customdata=textos_hover,
        hovertemplate="%{customdata}<br>Terminadas: %{y}<extra></extra>",
    )
    fig.add_scatter(
        x=etiquetas, y=[promedio] * len(y), mode="lines",
        name=f"Promedio ({promedio:.1f})",
        line=dict(color=COLOR_AMARILLO, width=3, dash="dash"),
        hovertemplate="Promedio: %{y:.1f}<extra></extra>",
    )
    media_movil = pd.Series(y).rolling(VENTANA_MEDIA_MOVIL, min_periods=1).mean()
    fig.add_scatter(
        x=etiquetas, y=media_movil, mode="lines+markers",
        name=f"Media móvil {VENTANA_MEDIA_MOVIL} semanas",
        line=dict(color=COLOR_GRIS, width=2),
        hovertemplate="Media móvil: %{y:.1f}<extra></extra>",
    )
    if len(y) >= 2:
        x_num = np.arange(len(y))
        pendiente, intercepto = np.polyfit(x_num, y, 1)
        fig.add_scatter(
            x=etiquetas, y=pendiente * x_num + intercepto, mode="lines",
            name=f"Tendencia ({pendiente:+.2f}/semana)",
            line=dict(color=COLOR_AZUL_CLARO, width=2, dash="dot"),
            hovertemplate="Tendencia: %{y:.1f}<extra></extra>",
        )
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white", legend=dict(orientation="h", y=-0.2),
        xaxis_title="Semana (inicia el lunes)", yaxis_title="Tareas terminadas",
        bargap=0.25,
    )
    fig.update_yaxes(gridcolor="#E5E8EF", rangemode="tozero")
    return fig


def pestana_historial(inicio: pd.Timestamp, fin: pd.Timestamp, semanas: int):
    st.subheader("Historial")
    datos = st.session_state.get("datos")
    if datos is None:
        st.info("Primero carga y valida un archivo en la pestaña **Cargar**.")
        return

    st.caption(
        f"Semanas completas de lunes a domingo, del **{fmt_fecha_larga(inicio)}** "
        f"al **{fmt_fecha_larga(fin)}** ({semanas} semanas). "
        "Puedes cambiar la fecha de corte y el número de semanas en la barra lateral."
    )

    # --- Filtros (acotan el análisis; no comparan personas) -------------------
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        rango = st.date_input(
            "Rango de fechas (dentro del historial)",
            value=(inicio.date(), fin.date()),
            min_value=inicio.date(), max_value=fin.date(), format="YYYY-MM-DD",
            key=f"rango_{inicio.date()}_{fin.date()}",
        )
    with f2:
        responsables = sorted(datos["responsable_n"].unique())
        sel_resp = st.multiselect(
            "Responsable", responsables, default=[],
            placeholder="Todos (equipo completo)",
        )
    with f3:
        sel_tipo = st.multiselect("Tipo", tipos_presentes(datos), default=[], placeholder="Todos")

    st.caption("Los filtros solo acotan el análisis del equipo: la app no muestra "
               "rankings ni comparaciones entre personas.")

    filtrados = datos
    if sel_resp:
        filtrados = filtrados[filtrados["responsable_n"].isin(sel_resp)]
    if sel_tipo:
        filtrados = filtrados[filtrados["tipo_n"].isin(sel_tipo)]

    tabla, excluidas = calcular_historial(filtrados, inicio, fin)

    # Rango de fechas: se muestran las semanas que se cruzan con el rango elegido
    if isinstance(rango, (list, tuple)) and len(rango) == 2:
        desde, hasta = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
        tabla = tabla[(tabla["fin_semana"] >= desde) & (tabla["inicio_semana"] <= hasta)]
    tabla = tabla.reset_index(drop=True)

    # --- Aviso de excluidas ---------------------------------------------------
    if len(excluidas):
        resumen = excluidas["motivo"].value_counts()
        detalle = "\n".join(f"- **{n}** · {m}" for m, n in resumen.items())
        st.warning(f"⚠️ Se excluyeron **{len(excluidas)}** tareas finalizadas del historial:\n\n{detalle}")
        with st.expander("Ver tareas excluidas"):
            st.dataframe(
                excluidas[["clave", "tipo_n", "fecha_fin", "motivo"]].rename(columns={"tipo_n": "tipo"}),
                hide_index=True, **ANCHO,
            )
    else:
        st.success("No se excluyó ninguna tarea finalizada.")

    if tabla.empty:
        st.info("No hay semanas en el rango seleccionado.")
        return

    # --- Indicadores ----------------------------------------------------------
    est = estadisticas(tabla)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tareas terminadas", est["total"])
    c2.metric("Promedio semanal", f"{est['promedio']:.1f}")
    c3.metric("Mediana semanal", f"{est['mediana']:.1f}")
    c4.metric("Semanas en cero", f"{est['semanas_cero']} de {est['semanas']}")
    c5.metric("Variabilidad (CV)", f"{est['variabilidad']:.2f}",
              help="Desviación estándar / promedio. Mayor a 0,5 indica alta variabilidad.")

    # --- Gráfico --------------------------------------------------------------
    st.plotly_chart(grafico_historial(tabla, est["promedio"]), **ANCHO)

    # --- Tabla semanal --------------------------------------------------------
    with st.expander("Ver tabla semanal"):
        vista = pd.DataFrame({
            "Semana": range(1, len(tabla) + 1),
            "Desde (lunes)": [fmt_fecha(d) for d in tabla["inicio_semana"]],
            "Hasta (domingo)": [fmt_fecha(d) for d in tabla["fin_semana"]],
            "Terminadas": tabla["terminadas"],
        })
        st.dataframe(vista, hide_index=True, **ANCHO)


# -----------------------------------------------------------------------------
# 9. INTERFAZ — PRONÓSTICO
# -----------------------------------------------------------------------------
def html_tarjeta(config, perc, pron, fecha_corte, fecha_comite) -> str:
    p, etiqueta, subtitulo, fondo, texto, acento = config
    insignia = ""
    if p == 0.85:
        insignia = (f'<span class="insignia" style="background:{COLOR_AZUL};'
                    f'color:{COLOR_AMARILLO};">★ USAR EN EL COMITÉ</span>')

    if pron["pendientes"] == 0:
        fecha_txt, dia_txt, semanas_txt = "Sin pendientes", "", "Nada por terminar"
        estado_comite = ""
    elif perc["sin_fecha"]:
        fecha_txt, dia_txt = "Sin fecha", ""
        semanas_txt = f"No se alcanza en {SEMANAS_TOPE} semanas"
        estado_comite = "✖ No llega a la fecha del comité"
    else:
        f = perc["fecha"]
        fecha_txt = fmt_fecha(f)
        dia_txt = DIAS[f.weekday()]
        semanas_txt = f"<b>{perc['semanas']}</b> semana{'s' if perc['semanas'] != 1 else ''} desde el corte"
        if f <= pd.Timestamp(fecha_comite):
            estado_comite = f"✔ Llega antes del {fmt_fecha(fecha_comite)}"
        else:
            dias_tarde = (f - pd.Timestamp(fecha_comite)).days
            estado_comite = f"✖ {dias_tarde} días después del {fmt_fecha(fecha_comite)}"

    posicion = math.ceil(p * SIMULACIONES)
    origen = (
        f"Sale de ordenar {fmt_num(SIMULACIONES, 0)} simulaciones y tomar la n.º "
        f"{fmt_num(posicion, 0)}: en el {etiqueta} de ellas el equipo terminó las "
        f"{pron['pendientes']} pendientes en {perc['semanas'] if not perc['sin_fecha'] else '—'} "
        f"semanas o menos, usando {pron['semanas_historial']} semanas de historial "
        f"(promedio {fmt_num(pron['promedio'])} tareas/semana)."
    )
    ancho = int(p * 100)
    return f"""
    <div class="tarjeta" style="background:{fondo}; color:{texto};">
      <div class="prob">{etiqueta}{insignia}</div>
      <div class="sub" style="color:{acento};">{subtitulo}</div>
      <div class="barra-prob"><div style="width:{ancho}%; background:{acento};"></div></div>
      <div class="fecha">{fecha_txt}</div>
      <div class="dia">{dia_txt}</div>
      <div class="semanas">{semanas_txt}</div>
      <div class="semanas" style="font-weight:700;">{estado_comite}</div>
      <div class="origen">{origen}</div>
    </div>
    """


def grafico_distribucion(pron: dict, fecha_corte) -> go.Figure:
    """Barras: en cuántas simulaciones se termina cada semana, con marcas 50/85/95."""
    resultados = pron["resultados"]
    validos = resultados[resultados <= SEMANAS_TOPE]
    conteo = pd.Series(validos).value_counts().sort_index()
    fechas = [fecha_fin_semana_n(fecha_corte, int(n)) for n in conteo.index]

    colores = []
    p50, p85, p95 = (x["semanas"] for x in pron["percentiles"])
    for n in conteo.index:
        if n <= p50:
            colores.append(COLOR_GRIS_CLARO)
        elif n <= p85:
            colores.append(COLOR_AMARILLO)
        elif n <= p95:
            colores.append(COLOR_AZUL)
        else:
            colores.append("#E5E8EF")

    fig = go.Figure()
    fig.add_bar(
        x=fechas, y=conteo.values, marker_color=colores,
        customdata=[[int(n), fmt_fecha(f)] for n, f in zip(conteo.index, fechas)],
        hovertemplate="Terminan el %{customdata[1]} (semana %{customdata[0]})"
                      "<br>%{y} simulaciones<extra></extra>",
    )
    for perc, (p, etiqueta, *_rest) in zip(pron["percentiles"], PERCENTILES):
        if perc["fecha"] is not None:
            fig.add_vline(x=perc["fecha"], line_width=2, line_dash="dash",
                          line_color=COLOR_AZUL if p != 0.85 else "#B89F00")
            fig.add_annotation(x=perc["fecha"], y=1.02, yref="paper", text=etiqueta,
                               showarrow=False, font=dict(color=COLOR_AZUL, size=12))
    fig.update_layout(
        title=dict(text="¿Qué semana se termina en cada simulación?", font=dict(size=15)),
        height=330, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white",
        showlegend=False, xaxis_title="Fecha de término (domingo)",
        yaxis_title="Simulaciones", bargap=0.1,
    )
    fig.update_yaxes(gridcolor="#E5E8EF")
    return fig


def grafico_probabilidad(pron: dict, fecha_corte, fecha_comite) -> go.Figure:
    """Curva acumulada: probabilidad de haber terminado en cada fecha."""
    resultados = pron["resultados"]
    max_n = int(min(resultados.max(), SEMANAS_TOPE))
    semanas = np.arange(1, max_n + 1)
    prob = np.array([(resultados <= n).mean() for n in semanas]) * 100
    fechas = [fecha_fin_semana_n(fecha_corte, int(n)) for n in semanas]

    fig = go.Figure()
    fig.add_scatter(
        x=fechas, y=prob, mode="lines", line=dict(color=COLOR_AZUL, width=3, shape="hv"),
        fill="tozeroy", fillcolor="rgba(15,38,92,0.08)",
        customdata=[fmt_fecha(f) for f in fechas],
        hovertemplate="Al %{customdata}: %{y:.0f} % de probabilidad<extra></extra>",
    )
    for perc, (p, etiqueta, *_rest) in zip(pron["percentiles"], PERCENTILES):
        if perc["fecha"] is not None:
            fig.add_scatter(
                x=[perc["fecha"]], y=[p * 100], mode="markers+text",
                marker=dict(size=13, color=COLOR_AMARILLO if p == 0.85 else COLOR_AZUL,
                            line=dict(color=COLOR_AZUL, width=2)),
                text=[f"{etiqueta} · {fmt_fecha(perc['fecha'])}"],
                textposition="middle left", textfont=dict(color=COLOR_AZUL, size=11),
                hoverinfo="skip",
            )
    comite = pd.Timestamp(fecha_comite)
    prob_comite = probabilidad_antes_de(resultados, fecha_corte, fecha_comite) * 100
    fig.add_vline(x=comite, line_color="#C0392B", line_width=2, line_dash="dot")
    fig.add_annotation(x=comite, y=prob_comite, text=f"Comité {fmt_fecha(comite)}: {prob_comite:.0f} %",
                       showarrow=True, arrowcolor="#C0392B", font=dict(color="#C0392B", size=12),
                       bgcolor="white", ax=60, ay=-30)
    fig.update_layout(
        title=dict(text="Probabilidad de haber terminado en cada fecha", font=dict(size=15)),
        height=330, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white",
        showlegend=False, xaxis_title="Fecha", yaxis_title="Probabilidad (%)",
    )
    fig.update_yaxes(range=[0, 105], gridcolor="#E5E8EF", ticksuffix=" %")
    return fig


def grafico_comparacion_tipos(filas: list, fecha_comite) -> go.Figure:
    """Una franja por tipo: de 50 % (izquierda) a 95 % (derecha), con el 85 % resaltado."""
    fig = go.Figure()
    etiquetas = [f["nombre"] for f in filas]
    for f in filas:
        if f["fechas"][0] is None or f["fechas"][2] is None:
            continue
        fig.add_scatter(
            x=[f["fechas"][0], f["fechas"][2]], y=[f["nombre"]] * 2, mode="lines",
            line=dict(color=COLOR_GRIS_CLARO, width=10), hoverinfo="skip", showlegend=False,
        )
    for idx, (p, etiqueta, *_rest) in enumerate(PERCENTILES):
        xs = [f["fechas"][idx] for f in filas]
        ys = [f["nombre"] for f, x in zip(filas, xs) if x is not None]
        xs = [x for x in xs if x is not None]
        fig.add_scatter(
            x=xs, y=ys, mode="markers", name=etiqueta,
            marker=dict(size=16 if p == 0.85 else 12,
                        color={0.5: COLOR_GRIS, 0.85: COLOR_AMARILLO, 0.95: COLOR_AZUL}[p],
                        line=dict(color=COLOR_AZUL, width=1.5),
                        symbol="diamond" if p == 0.85 else "circle"),
            customdata=[fmt_fecha(x) for x in xs],
            hovertemplate=f"{etiqueta}: " + "%{customdata}<extra>%{y}</extra>",
        )
    fig.add_vline(x=pd.Timestamp(fecha_comite), line_color="#C0392B", line_dash="dot", line_width=2)
    fig.update_layout(
        height=90 + 55 * len(filas), margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white", legend=dict(orientation="h", y=1.15, x=0),
        xaxis_title="Fecha de término", yaxis=dict(categoryorder="array",
                                                   categoryarray=list(reversed(etiquetas))),
    )
    fig.update_xaxes(gridcolor="#E5E8EF")
    return fig


def pestana_pronostico(inicio, fin, semanas, fecha_corte, fecha_comite):
    st.subheader("Pronóstico")
    datos = st.session_state.get("datos")
    if datos is None:
        st.info("Primero carga y valida un archivo en la pestaña **Cargar**.")
        return

    tipos = tipos_presentes(datos)
    opciones = ["Todos"] + tipos
    sel = st.radio("Tipo de tarea", opciones, horizontal=True, key="tipo_pronostico",
                   help="Cada tipo usa su propio historial y sus propias pendientes.")
    subconjunto = datos if sel == "Todos" else datos[datos["tipo_n"] == sel]

    with st.spinner(f"Ejecutando {fmt_num(SIMULACIONES, 0)} simulaciones…"):
        pron = pronosticar(subconjunto, inicio, fin, fecha_corte)

    # --- Contexto -------------------------------------------------------------
    prob_comite = probabilidad_antes_de(pron["resultados"], fecha_corte, fecha_comite)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pendientes", pron["pendientes"],
              help="Todas las tareas cuyo estado no es Finalizada.")
    c2.metric("En curso / Por hacer", f"{pron['en_curso']} / {pron['por_hacer']}")
    c3.metric("Ritmo histórico", f"{fmt_num(pron['promedio'])} /semana",
              help=f"Promedio de las {pron['semanas_historial']} semanas de historial.")
    c4.metric(f"Prob. de terminar al {fmt_fecha(fecha_comite)}",
              f"{prob_comite * 100:.0f} %" if pron["pendientes"] else "—")
    if pron["otros"]:
        st.caption(f"Incluye {pron['otros']} pendientes con estado no reconocido.")

    seccion_cuando(datos, pron, sel, tipos, inicio, fin, fecha_corte, fecha_comite)

    # --- v4: ¿Cuántas tareas alcanzamos antes de…? -----------------------------
    st.divider()
    seccion_cuantas(subconjunto, sel, inicio, fin, fecha_corte)


def seccion_cuando(datos, pron, sel, tipos, inicio, fin, fecha_corte, fecha_comite):
    """¿Cuándo terminamos las pendientes? (v3)"""
    if pron["pendientes"] == 0:
        st.success(f"No hay tareas pendientes{'' if sel == 'Todos' else f' de tipo {sel}'}.")
        return
    if sum(pron["throughput"]) == 0:
        st.error(
            "No se puede pronosticar: en el historial elegido no hay ninguna tarea "
            f"{'' if sel == 'Todos' else f'de tipo {sel} '}terminada. "
            "Aumenta las semanas de historial o revisa el archivo."
        )
        return

    # --- Tarjetas -------------------------------------------------------------
    st.markdown(
        f"#### ¿Cuándo terminamos las **{pron['pendientes']}** pendientes"
        f"{'' if sel == 'Todos' else f' de tipo {sel}'}?"
    )
    columnas = st.columns(3, gap="medium")
    for col, config, perc in zip(columnas, PERCENTILES, pron["percentiles"]):
        with col:
            st.markdown(html_tarjeta(config, perc, pron, fecha_corte, fecha_comite),
                        unsafe_allow_html=True)

    st.caption(
        f"Semana 1 = del {fmt_fecha(lunes_de(fecha_corte))} al "
        f"{fmt_fecha_larga(fecha_fin_semana_n(fecha_corte, 1))}. "
        f"Simulación con semilla fija ({SEMILLA}): el resultado es el mismo cada vez que se abre."
    )
    if any(p["sin_fecha"] for p in pron["percentiles"]):
        st.warning(f"Algunas simulaciones no terminan en {SEMANAS_TOPE} semanas: "
                   "el ritmo histórico es muy bajo frente a las pendientes.")

    # --- Gráficos de apoyo ----------------------------------------------------
    st.write("")
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(grafico_distribucion(pron, fecha_corte), **ANCHO)
        st.caption("Cada barra es una semana de término. Gris = hasta el 50 %, "
                   "amarillo = hasta el 85 %, azul = hasta el 95 %.")
    with g2:
        st.plotly_chart(grafico_probabilidad(pron, fecha_corte, fecha_comite), **ANCHO)
        st.caption("Cuanto más a la derecha, más seguro. La línea roja es la fecha del comité.")

    # --- Comparación rápida entre tipos ---------------------------------------
    st.markdown("#### Comparación rápida por tipo")
    filas = []
    for nombre in ["Todos"] + tipos:
        sub = datos if nombre == "Todos" else datos[datos["tipo_n"] == nombre]
        pr = pron if nombre == sel else pronosticar(sub, inicio, fin, fecha_corte)
        sin_hist = pr["pendientes"] > 0 and sum(pr["throughput"]) == 0
        filas.append({
            "nombre": nombre,
            "pendientes": pr["pendientes"],
            "promedio": pr["promedio"],
            "fechas": [None if (sin_hist or pr["pendientes"] == 0) else p["fecha"]
                       for p in pr["percentiles"]],
            "prob": probabilidad_antes_de(pr["resultados"], fecha_corte, fecha_comite)
                    if pr["pendientes"] and not sin_hist else None,
            "sin_hist": sin_hist,
        })

    tabla = pd.DataFrame({
        "Tipo": [f["nombre"] for f in filas],
        "Pendientes": [f["pendientes"] for f in filas],
        "Ritmo /semana": [fmt_num(f["promedio"]) for f in filas],
        "50 %": [fmt_fecha(f["fechas"][0]) if f["fechas"][0] is not None else "—" for f in filas],
        "85 % ★": [fmt_fecha(f["fechas"][1]) if f["fechas"][1] is not None else "—" for f in filas],
        "95 %": [fmt_fecha(f["fechas"][2]) if f["fechas"][2] is not None else "—" for f in filas],
        f"Prob. al {fmt_fecha(fecha_comite)}": [
            f"{f['prob'] * 100:.0f} %" if f["prob"] is not None
            else ("sin historial" if f["sin_hist"] else "sin pendientes") for f in filas],
    })
    t1, t2 = st.columns([3, 2])
    with t1:
        st.plotly_chart(grafico_comparacion_tipos(filas, fecha_comite), **ANCHO)
    with t2:
        st.dataframe(tabla, hide_index=True, **ANCHO)
    st.caption(
        "Cada tipo se simula por separado con su propio historial. Por eso las fechas por tipo "
        "no se suman: sirven para ver qué tipo de trabajo va más atrasado."
    )


def html_tarjeta_alcance(config, esc, res, fecha_limite) -> str:
    p, etiqueta, subtitulo, fondo, texto, acento = config
    insignia = ""
    if p == 0.85:
        insignia = (f'<span class="insignia" style="background:{COLOR_AZUL};'
                    f'color:{COLOR_AMARILLO};">★ USAR EN EL COMITÉ</span>')
    cantidad = esc["cantidad"]
    pend = res["pendientes"]
    if pend:
        cubre = min(cantidad, pend)
        avance = f"Cubre <b>{cubre}</b> de {pend} pendientes ({cubre / pend * 100:.0f} %)"
        ancho = int(min(1, cantidad / pend) * 100)
        estado = ("✔ Alcanza para todas las pendientes" if cantidad >= pend
                  else f"✖ Habría que sacar {pend - cantidad} del alcance")
    else:
        avance, ancho, estado = "No hay pendientes", 100, ""
    pos = esc["posicion"]
    lugar = "la mitad" if p == 0.50 else f"la posición {100 - int(p * 100)} %"
    origen = (
        f"Sale de sumar {res['semanas']} semanas sacadas al azar del historial "
        f"({res['semanas_historial']} semanas, promedio {fmt_num(res['promedio'])}/semana), "
        f"{fmt_num(SIMULACIONES, 0)} veces. Se ordenan las sumas de menor a mayor y se toma "
        f"{lugar} (n.º {fmt_num(pos, 0)}): en el {etiqueta} de las simulaciones se terminaron "
        f"{cantidad} tareas o más."
    )
    return f"""
    <div class="tarjeta" style="background:{fondo}; color:{texto};">
      <div class="prob">{etiqueta}{insignia}</div>
      <div class="sub" style="color:{acento};">{subtitulo}</div>
      <div class="dia" style="margin-top:.8rem;">al menos</div>
      <div class="fecha" style="font-size:3rem; margin-top:0;">{cantidad}</div>
      <div class="dia">tareas terminadas hasta el {fmt_fecha_larga(fecha_limite)}</div>
      <div class="barra-prob"><div style="width:{ancho}%; background:{acento};"></div></div>
      <div class="semanas">{avance}</div>
      <div class="semanas" style="font-weight:700;">{estado}</div>
      <div class="origen">{origen}</div>
    </div>
    """


def grafico_sumas(res: dict) -> go.Figure:
    """Barras con las 10.000 sumas: cuántas simulaciones dieron cada total."""
    sumas = res["sumas"]
    conteo = pd.Series(sumas).value_counts().sort_index()
    c50, c85, c95 = (e["cantidad"] for e in res["escenarios"])
    colores = []
    for v in conteo.index:
        if v < c95:
            colores.append("#E5E8EF")          # el 5 % más pesimista
        elif v < c85:
            colores.append(COLOR_AZUL)         # entre 95 % y 85 %
        elif v < c50:
            colores.append(COLOR_AMARILLO)     # entre 85 % y 50 %
        else:
            colores.append(COLOR_GRIS_CLARO)   # mitad optimista
    fig = go.Figure()
    fig.add_bar(
        x=conteo.index, y=conteo.values, marker_color=colores,
        marker_line=dict(color="white", width=0.5),
        customdata=[(sumas >= v).mean() * 100 for v in conteo.index],
        hovertemplate="Suma: %{x} tareas<br>%{y} simulaciones"
                      "<br>%{customdata:.0f} % terminó esta cantidad o más<extra></extra>",
    )
    for esc, (p, etiqueta, *_r) in zip(res["escenarios"], ESCENARIOS_ALCANCE):
        fig.add_vline(x=esc["cantidad"], line_width=2, line_dash="dash",
                      line_color="#B89F00" if p == 0.85 else COLOR_AZUL)
        fig.add_annotation(x=esc["cantidad"], y=1.04, yref="paper", showarrow=False,
                           text=f"<b>{etiqueta}</b>: ≥ {esc['cantidad']}",
                           font=dict(color=COLOR_AZUL, size=12), bgcolor="white")
    if res["pendientes"]:
        fig.add_vline(x=res["pendientes"], line_color="#C0392B", line_dash="dot", line_width=2)
        fig.add_annotation(x=res["pendientes"], y=0.88, yref="paper", showarrow=False,
                           text=f"Pendientes: {res['pendientes']}", bgcolor="white",
                           font=dict(color="#C0392B", size=12))
    fig.update_layout(
        title=dict(text=f"Las {fmt_num(SIMULACIONES, 0)} sumas: tareas terminadas en "
                        f"{res['semanas']} semanas", font=dict(size=15)),
        height=380, margin=dict(l=10, r=10, t=60, b=10), plot_bgcolor="white",
        showlegend=False, xaxis_title="Tareas terminadas (suma de la simulación)",
        yaxis_title="Simulaciones", bargap=0.08,
    )
    fig.update_yaxes(gridcolor="#E5E8EF")
    return fig


def grafico_escenarios_alcance(res: dict) -> go.Figure:
    """Barras horizontales: 50 / 85 / 95 % frente a las pendientes."""
    etiquetas = [f"{c[1]} · {c[2]}" for c in ESCENARIOS_ALCANCE]
    valores = [e["cantidad"] for e in res["escenarios"]]
    fig = go.Figure()
    fig.add_bar(
        y=etiquetas, x=valores, orientation="h",
        marker_color=[COLOR_GRIS_CLARO, COLOR_AMARILLO, COLOR_AZUL],
        marker_line=dict(color=COLOR_AZUL, width=1.5),
        text=[f"al menos {v}" for v in valores], textposition="outside",
        hovertemplate="%{y}: al menos %{x} tareas<extra></extra>",
    )
    if res["pendientes"]:
        fig.add_vline(x=res["pendientes"], line_color="#C0392B", line_dash="dot", line_width=2)
        fig.add_annotation(x=res["pendientes"], y=1.1, yref="paper", showarrow=False,
                           text=f"Pendientes: {res['pendientes']}",
                           font=dict(color="#C0392B", size=12))
    tope = max(valores + [res["pendientes"]]) * 1.25 + 1
    fig.update_layout(
        title=dict(text="Compromiso posible frente a las pendientes", font=dict(size=15)),
        height=380, margin=dict(l=10, r=10, t=60, b=10), plot_bgcolor="white",
        showlegend=False, xaxis=dict(range=[0, tope], title="Tareas"),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(etiquetas))),
    )
    fig.update_xaxes(gridcolor="#E5E8EF")
    return fig


def seccion_cuantas(subconjunto, sel, inicio, fin, fecha_corte):
    """v4 — ¿Cuántas tareas alcanzamos antes de…? (la fecha no se mueve; se ajusta el alcance)."""
    sufijo = "" if sel == "Todos" else f" de tipo {sel}"
    st.markdown(f"### 🎯 ¿Cuántas tareas{sufijo} alcanzamos antes de…?")
    st.caption("Úsalo cuando la fecha no se mueve y lo que se ajusta es el alcance.")

    c_fecha, c_info = st.columns([1, 3])
    with c_fecha:
        fecha_limite = st.date_input(
            "Fecha límite", value=FECHA_LIMITE_DEFECTO, format="YYYY-MM-DD",
            min_value=fecha_corte, key="fecha_limite",
            help="Se cuentan las semanas completas (lunes a domingo) desde la fecha de corte.",
        )

    res = pronosticar_alcance(subconjunto, inicio, fin, fecha_corte, fecha_limite)

    with c_info:
        a, b, c = st.columns(3)
        a.metric("Semanas disponibles", res["semanas"],
                 help=f"Del {fmt_fecha(lunes_de(fecha_corte))} al "
                      f"{fmt_fecha(res['ultimo_domingo']) if res['ultimo_domingo'] is not None else '—'}.")
        b.metric("Pendientes" + ("" if sel == "Todos" else f" ({sel})"), res["pendientes"])
        c.metric("Ritmo histórico", f"{fmt_num(res['promedio'])} /semana",
                 help=f"Promedio de las {res['semanas_historial']} semanas de historial.")

    if res["semanas"] == 0:
        st.warning(f"Entre el {fmt_fecha(fecha_corte)} y el {fmt_fecha(fecha_limite)} no hay ninguna "
                   "semana completa (lunes a domingo). Elige una fecha límite posterior.")
        return
    if sum(res["throughput"]) == 0:
        st.error(f"No se puede pronosticar: en el historial no hay ninguna tarea{sufijo} terminada. "
                 "Aumenta las semanas de historial o revisa el archivo.")
        return
    if res["ultimo_domingo"].normalize() != pd.Timestamp(fecha_limite).normalize():
        st.info(f"La fecha límite cae a mitad de semana: se cuentan {res['semanas']} semanas completas, "
                f"hasta el {fmt_fecha_larga(res['ultimo_domingo'])}.")

    # --- Respuesta en una línea ----------------------------------------------
    linea = " · ".join(f"{c[1]}: al menos <b>{e['cantidad']}</b>"
                       for c, e in zip(ESCENARIOS_ALCANCE, res["escenarios"]))
    st.markdown(
        f"<div class='caja-requisitos' style='font-size:1.1rem;'>Antes del "
        f"<b>{fmt_fecha_larga(fecha_limite)}</b> el equipo termina → {linea} tareas</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # --- Tarjetas -------------------------------------------------------------
    columnas = st.columns(3, gap="medium")
    for col, config, esc in zip(columnas, ESCENARIOS_ALCANCE, res["escenarios"]):
        with col:
            st.markdown(html_tarjeta_alcance(config, esc, res, fecha_limite), unsafe_allow_html=True)
    st.caption(f"Simulación con semilla fija ({SEMILLA}): el resultado es el mismo cada vez que se abre.")

    # --- Gráficos -------------------------------------------------------------
    st.write("")
    g1, g2 = st.columns([3, 2])
    with g1:
        st.plotly_chart(grafico_sumas(res), **ANCHO)
        st.caption("Cada barra es un total posible y su altura, cuántas de las 10.000 simulaciones "
                   "lo obtuvieron. Gris claro = mitad optimista · amarillo = hasta el 85 % · "
                   "azul = hasta el 95 % · gris tenue = el 5 % más pesimista.")
    with g2:
        st.plotly_chart(grafico_escenarios_alcance(res), **ANCHO)
        st.caption("Si una barra pasa la línea roja, ese escenario alcanza para todas las pendientes. "
                   "Si no, la diferencia es lo que habría que sacar del alcance.")


# -----------------------------------------------------------------------------
# 9b. INTERFAZ — TIEMPO DE CICLO (v5)
# -----------------------------------------------------------------------------
def txt_dias(n) -> str:
    return f"{n} día" if n == 1 else f"{n} días"


def html_tarjeta_ciclo(titulo, subtitulo, valor, unidad, fecha_lista, fondo, texto, acento,
                       origen, insignia="") -> str:
    fecha_html = (f'<div class="semanas">Si empieza hoy, lista el<br><b>{fmt_fecha_larga(fecha_lista)}</b></div>'
                  if fecha_lista is not None else "")
    return f"""
    <div class="tarjeta" style="background:{fondo}; color:{texto};">
      <div class="prob" style="font-size:2rem;">{titulo}{insignia}</div>
      <div class="sub" style="color:{acento};">{subtitulo}</div>
      <div class="fecha" style="font-size:3rem; margin-top:.6rem;">{valor}</div>
      <div class="dia">{unidad}</div>
      {fecha_html}
      <div class="origen">{origen}</div>
    </div>
    """


def grafico_dias(dias, res: dict) -> go.Figure:
    """Barras: cuántas tareas tardaron 0, 1, 2, 3… días, con marcas de promedio, 50 % y 85 %."""
    conteo = pd.Series(dias).value_counts().sort_index()
    eje = np.arange(conteo.index.min(), conteo.index.max() + 1)
    conteo = conteo.reindex(eje, fill_value=0)
    colores = [COLOR_AMARILLO if d <= res["p50"] else COLOR_AZUL if d <= res["p85"] else COLOR_GRIS_CLARO
               for d in conteo.index]
    acumulado = conteo.cumsum() / conteo.sum() * 100
    fig = go.Figure()
    fig.add_bar(
        x=conteo.index, y=conteo.values, marker_color=colores,
        marker_line=dict(color=COLOR_AZUL, width=0.6),
        customdata=acumulado.values,
        hovertemplate="%{x} días: %{y} tareas<br>%{customdata:.0f} % tardó esto o menos<extra></extra>",
    )
    marcas = [
        (res["p50"], f"50 %: {txt_dias(res['p50'])}", "#B89F00", "dash", 1.10),
        (res["p85"], f"85 %: {txt_dias(res['p85'])}", COLOR_AZUL, "dash", 1.10),
        (res["promedio"], f"Promedio: {fmt_num(res['promedio'])}", "#C0392B", "dot", 1.02),
    ]
    for x, texto, color, estilo, y in marcas:
        fig.add_vline(x=x, line_color=color, line_dash=estilo, line_width=2)
        fig.add_annotation(x=x, y=y, yref="paper", text=f"<b>{texto}</b>", showarrow=False,
                           font=dict(color=color, size=12), bgcolor="white")
    fig.update_layout(
        title=dict(text="¿Cuántas tareas tardaron 0, 1, 2, 3… días?", font=dict(size=15)),
        height=400, margin=dict(l=10, r=10, t=75, b=10), plot_bgcolor="white",
        showlegend=False, xaxis_title="Días de calendario (fecha_fin − fecha_inicio)",
        yaxis_title="Tareas", bargap=0.12,
    )
    fig.update_xaxes(dtick=1 if len(eje) <= 40 else None)
    fig.update_yaxes(gridcolor="#E5E8EF")
    return fig


def grafico_ciclo_por_tipo(filas: list) -> go.Figure:
    """Barras agrupadas sencillas: 50 % y 85 % por tipo."""
    nombres = [f["Tipo"] for f in filas]
    fig = go.Figure()
    fig.add_bar(name="50 %", x=nombres, y=[f["_p50"] for f in filas], marker_color=COLOR_AMARILLO,
                marker_line=dict(color=COLOR_AZUL, width=1), text=[f["_p50"] for f in filas],
                textposition="outside")
    fig.add_bar(name="85 %", x=nombres, y=[f["_p85"] for f in filas], marker_color=COLOR_AZUL,
                text=[f["_p85"] for f in filas], textposition="outside")
    fig.update_layout(
        barmode="group", height=320, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
        legend=dict(orientation="h", y=1.15, x=0), yaxis_title="Días o menos",
    )
    fig.update_yaxes(gridcolor="#E5E8EF", rangemode="tozero")
    return fig


def pestana_tiempo_ciclo(inicio, fin, semanas, fecha_corte):
    st.subheader("Tiempo de ciclo")
    datos = st.session_state.get("datos")
    if datos is None:
        st.info("Primero carga y valida un archivo en la pestaña **Cargar**.")
        return

    st.markdown("### ⏱️ Si empezamos esta tarea hoy, ¿cuándo estará lista?")
    st.caption(
        f"Basado en las tareas terminadas entre el **{fmt_fecha_larga(inicio)}** y el "
        f"**{fmt_fecha_larga(fin)}** ({semanas} semanas de historial). «Hoy» = fecha de corte "
        f"({fmt_fecha(fecha_corte)}); ambas se cambian en la barra lateral."
    )

    tipos = tipos_presentes(datos)
    sel = st.radio("Tipo de tarea", ["Todos"] + tipos, horizontal=True, key="tipo_ciclo",
                   help="Cada tipo de tarea puede tardar distinto.")
    subconjunto = datos if sel == "Todos" else datos[datos["tipo_n"] == sel]
    sufijo = "" if sel == "Todos" else f" de tipo {sel}"

    validas, excluidas = calcular_tiempo_ciclo(subconjunto, inicio, fin)
    res = resumen_ciclo(validas["dias"])

    # --- Aviso de excluidas ---------------------------------------------------
    if len(excluidas):
        detalle = "\n".join(f"- **{n}** · {m}" for m, n in excluidas["motivo"].value_counts().items())
        st.warning(f"⚠️ No se usaron **{len(excluidas)}** tareas finalizadas{sufijo}:\n\n{detalle}")
        with st.expander("Ver tareas no usadas"):
            st.dataframe(excluidas[["clave", "tipo_n", "fecha_inicio", "fecha_fin", "motivo"]]
                         .rename(columns={"tipo_n": "tipo"}), hide_index=True, **ANCHO)

    if res["n"] == 0:
        st.error(f"No hay tareas{sufijo} válidas en el historial para calcular el tiempo de ciclo. "
                 "Aumenta las semanas de historial o revisa las fechas del archivo.")
        return

    # --- Frase resumen --------------------------------------------------------
    st.markdown(
        f"<div class='caja-requisitos' style='font-size:1.25rem; line-height:1.5;'>"
        f"El <b>50 %</b> de las tareas{sufijo} se termina en <b>{txt_dias(res['p50'])} o menos</b>, "
        f"y el <b>85 %</b> en <b>{txt_dias(res['p85'])} o menos</b>.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # --- Tarjetas -------------------------------------------------------------
    hoy = pd.Timestamp(fecha_corte)
    n = res["n"]
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(html_tarjeta_ciclo(
            "Promedio", "Referencia", fmt_num(res["promedio"]), "días por tarea", None,
            "#FFFFFF", COLOR_AZUL, COLOR_GRIS,
            f"Suma de los días de las {n} tareas válidas dividida para {n}. "
            f"Rango: de {txt_dias(res['minimo'])} a {txt_dias(res['maximo'])}. "
            "Ojo: unas pocas tareas muy largas lo suben; para comprometer usa el 85 %."
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(html_tarjeta_ciclo(
            "50 %", "Probable", res["p50"], f"{'día' if res['p50'] == 1 else 'días'} o menos",
            hoy + pd.Timedelta(days=res["p50"]), COLOR_AMARILLO, COLOR_AZUL, COLOR_AZUL,
            f"Se ordenan los {n} tiempos de menor a mayor y se toma el de la posición 50 % "
            f"(n.º {max(1, math.ceil(0.5 * n))} de {n}): la mitad de las tareas tardó eso o menos."
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(html_tarjeta_ciclo(
            "85 %", "Recomendado", res["p85"], f"{'día' if res['p85'] == 1 else 'días'} o menos",
            hoy + pd.Timedelta(days=res["p85"]), COLOR_AZUL, "#FFFFFF", COLOR_AMARILLO,
            f"Se ordenan los {n} tiempos de menor a mayor y se toma el de la posición 85 % "
            f"(n.º {max(1, math.ceil(0.85 * n))} de {n}): 85 de cada 100 tareas tardaron eso o menos.",
            insignia=(f'<span class="insignia" style="background:{COLOR_AMARILLO};'
                      f'color:{COLOR_AZUL};">★ PARA COMPROMETER</span>'),
        ), unsafe_allow_html=True)
    st.caption("Días de calendario entre fecha_inicio y fecha_fin (solo las fechas, sin horas). "
               "0 días = empezó y terminó el mismo día.")

    # --- Gráfico de barras ----------------------------------------------------
    st.write("")
    st.plotly_chart(grafico_dias(validas["dias"], res), **ANCHO)
    st.caption("Amarillo = el 50 % más rápido · azul = hasta el 85 % · gris = el 15 % más lento.")

    # --- Comparación rápida por tipo ------------------------------------------
    st.markdown("#### Comparación rápida por tipo")
    filas = []
    for nombre in ["Todos"] + tipos:
        sub = datos if nombre == "Todos" else datos[datos["tipo_n"] == nombre]
        v, _ = calcular_tiempo_ciclo(sub, inicio, fin)
        r = resumen_ciclo(v["dias"])
        if r["n"] == 0:
            continue
        filas.append({
            "Tipo": nombre, "Tareas": r["n"], "Promedio (días)": fmt_num(r["promedio"]),
            "50 % (días o menos)": r["p50"], "85 % (días o menos)": r["p85"],
            "Lista al 85 % si empieza hoy": fmt_fecha(hoy + pd.Timedelta(days=r["p85"])),
            "_p50": r["p50"], "_p85": r["p85"],
        })
    t1, t2 = st.columns([2, 3])
    with t1:
        st.plotly_chart(grafico_ciclo_por_tipo(filas), **ANCHO)
    with t2:
        st.dataframe(pd.DataFrame(filas).drop(columns=["_p50", "_p85"]), hide_index=True, **ANCHO)


def pestana_pendiente(nombre: str):
    st.subheader(nombre)
    st.info("Disponible en la siguiente versión.")


# -----------------------------------------------------------------------------
# 10. PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITULO, page_icon="📈", layout="wide")
    aplicar_estilo()
    st.title(APP_TITULO)
    st.caption(f"Pronóstico de entregas basado en el historial del equipo · {APP_VERSION}")

    fecha_corte, semanas, inicio, fin, fecha_comite = panel_parametros()

    tabs = st.tabs(["Cargar", "Historial", "Pronóstico", "Tiempo de ciclo", "Informe"])
    with tabs[0]:
        pestana_cargar()
    with tabs[1]:
        pestana_historial(inicio, fin, semanas)
    with tabs[2]:
        pestana_pronostico(inicio, fin, semanas, fecha_corte, fecha_comite)
    with tabs[3]:
        pestana_tiempo_ciclo(inicio, fin, semanas, fecha_corte)
    with tabs[4]:
        pestana_pendiente("Informe")


if __name__ == "__main__":
    main()
