import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import sqlite3
import os
import tempfile
from datetime import datetime

try:
    from fpdf import FPDF
except ModuleNotFoundError:
    FPDF = None

# ==================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==================================================
st.set_page_config(page_title="Visualizador Topográfico", layout="wide")
st.title("🌎 Visualizador de Datos Topográficos")

# ==================================================
# INICIALIZACIÓN DE ESTADO
# ==================================================
if "tin_fig" not in st.session_state:
    st.session_state.tin_fig = None

if "parametros_diseno" not in st.session_state:
    st.session_state.parametros_diseno = None

if "eje_vial" not in st.session_state:
    st.session_state.eje_vial = None

if "rasante" not in st.session_state:
    st.session_state.rasante = None

if "pendientes_tramos" not in st.session_state:
    st.session_state.pendientes_tramos = {}

if "longitud_total" not in st.session_state:
    st.session_state.longitud_total = None

if "excavacion" not in st.session_state:
    st.session_state.excavacion = None

if "pdf_memoria" not in st.session_state:
    st.session_state.pdf_memoria = None

# ==================================================
# BASE DE DATOS
# ==================================================
DB_PATH = "diseños_viales.db"


def inicializar_base_datos():
    conexion = sqlite3.connect(DB_PATH)
    cursor = conexion.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS simulaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            ancho REAL,
            presupuesto REAL,
            longitud_lograda REAL,
            corte REAL,
            relleno REAL,
            fecha TEXT
        )
    """)

    conexion.commit()
    conexion.close()


def guardar_simulacion(nombre, ancho, presupuesto, longitud, corte, relleno):
    inicializar_base_datos()

    conexion = sqlite3.connect(DB_PATH)
    cursor = conexion.cursor()

    cursor.execute("""
        INSERT INTO simulaciones (
            nombre, ancho, presupuesto, longitud_lograda, corte, relleno, fecha
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        nombre,
        ancho,
        presupuesto,
        longitud,
        corte,
        relleno,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conexion.commit()
    conexion.close()


def leer_simulaciones():
    inicializar_base_datos()

    conexion = sqlite3.connect(DB_PATH)
    df_simulaciones = pd.read_sql_query(
        "SELECT * FROM simulaciones ORDER BY id DESC",
        conexion
    )
    conexion.close()

    return df_simulaciones


# ==================================================
# MENÚ LATERAL
# ==================================================
st.sidebar.title("Menú")
opcion = st.sidebar.radio(
    "Seleccione una opción",
    [
        "Estadísticas",
        "Nube de Puntos 3D",
        "Esqueleto Estructural (TIN)",
        "Superficie Sólida (MDE)",
        "Maqueta Topográfica (Bloque 3D)",
        "Parámetros de Diseño",
        "Diseño de Eje y Rasante",
        "Maqueta de Excavación 3D",
        "Base de Datos (Archivero)",
        "Emisión de Memoria (PDF)"
    ]
)

# ==================================================
# FUNCIONES AUXILIARES
# ==================================================
def formato_km(distancia):
    km = int(distancia // 1000)
    m = int(round(distancia % 1000))
    return f"K{km}+{m:03d}"


def calcular_distancias_acumuladas(eje_xy):
    eje_xy = np.array(eje_xy)
    distancias = [0]

    for i in range(1, len(eje_xy)):
        d = np.linalg.norm(eje_xy[i] - eje_xy[i - 1])
        distancias.append(distancias[-1] + d)

    return np.array(distancias)


def interpolar_eje(eje_xy, estaciones):
    eje_xy = np.array(eje_xy)
    distancias = calcular_distancias_acumuladas(eje_xy)

    x = np.interp(estaciones, distancias, eje_xy[:, 0])
    y = np.interp(estaciones, distancias, eje_xy[:, 1])

    return np.column_stack([x, y])


def crear_interpoladores_terreno(df):
    puntos = df[["X", "Y", "Z"]].values

    interp_lineal = LinearNDInterpolator(
        puntos[:, :2],
        puntos[:, 2]
    )

    interp_cercano = NearestNDInterpolator(
        puntos[:, :2],
        puntos[:, 2]
    )

    return interp_lineal, interp_cercano


def obtener_z_terreno(xy, interp_lineal, interp_cercano):
    z = interp_lineal(xy[:, 0], xy[:, 1])
    z = np.array(z, dtype=float)

    faltantes = np.isnan(z)

    if np.any(faltantes):
        z[faltantes] = interp_cercano(
            xy[faltantes, 0],
            xy[faltantes, 1]
        )

    return z


def agregar_curvas_superficie(fig, puntos, tri, z_min, z_max, color="white", ancho=5):
    intervalo = 5

    niveles = np.arange(
        np.floor(z_min / intervalo) * intervalo,
        np.ceil(z_max / intervalo) * intervalo + intervalo,
        intervalo
    )

    elevacion_visual = max((z_max - z_min) * 0.004, 0.05)

    for nivel in niveles:
        xs, ys, zs = [], [], []

        for simplex in tri.simplices:
            vertices = puntos[simplex]
            intersecciones = []

            for a, b in [(0, 1), (1, 2), (2, 0)]:
                p1 = vertices[a]
                p2 = vertices[b]
                z1 = p1[2]
                z2 = p2[2]

                if (z1 < nivel < z2) or (z2 < nivel < z1):
                    t = (nivel - z1) / (z2 - z1)
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    z = nivel + elevacion_visual
                    intersecciones.append([x, y, z])

            if len(intersecciones) == 2:
                xs += [intersecciones[0][0], intersecciones[1][0], None]
                ys += [intersecciones[0][1], intersecciones[1][1], None]
                zs += [intersecciones[0][2], intersecciones[1][2], None]

        if xs:
            fig.add_trace(go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=color, width=ancho),
                showlegend=False,
                hovertemplate=f"Curva de nivel: {nivel:.0f} m<extra></extra>"
            ))


def generar_eje_vial_automatico(df, pendiente_maxima=0.10):
    puntos_2d = df[["X", "Y"]].values

    idx_inicio = df["Z"].idxmin()
    idx_final = df["Z"].idxmax()

    inicio = df.loc[idx_inicio, ["X", "Y", "Z"]].astype(float).values
    final = df.loc[idx_final, ["X", "Y", "Z"]].astype(float).values

    inicio_xy = inicio[:2]
    final_xy = final[:2]

    vector = final_xy - inicio_xy
    distancia_directa = np.linalg.norm(vector)

    if distancia_directa == 0:
        raise ValueError("No se puede generar el eje porque el punto inicial y final coinciden.")

    direccion = vector / distancia_directa
    perpendicular = np.array([-direccion[1], direccion[0]])

    desnivel = final[2] - inicio[2]
    longitud_minima = abs(desnivel) / pendiente_maxima
    longitud_objetivo = max(distancia_directa * 1.15, longitud_minima * 1.05)

    rango_xy = max(
        df["X"].max() - df["X"].min(),
        df["Y"].max() - df["Y"].min()
    )

    tri = Delaunay(puntos_2d)

    mejor_eje = None
    mejor_score = np.inf

    t = np.linspace(0, 1, 500)
    base = inicio_xy + np.outer(t, vector)
    envolvente = np.sin(np.pi * t)

    amplitudes = np.linspace(0.03, 0.32, 12) * rango_xy
    ondas_posibles = [1, 2, 3, 4, 5]

    for ondas in ondas_posibles:
        for amplitud in amplitudes:
            offset = amplitud * np.sin(2 * np.pi * ondas * t) * envolvente
            eje_xy = base + np.outer(offset, perpendicular)

            dentro = tri.find_simplex(eje_xy) >= 0
            porcentaje_fuera = 1 - np.mean(dentro)

            longitud = calcular_distancias_acumuladas(eje_xy)[-1]
            pendiente_media = abs(desnivel) / longitud if longitud > 0 else np.inf

            exceso_pendiente = max(0, pendiente_media - pendiente_maxima)
            falta_longitud = max(0, longitud_objetivo - longitud)

            cambios = np.diff(eje_xy, axis=0)
            angulos = np.arctan2(cambios[:, 1], cambios[:, 0])
            suavidad = np.mean(np.abs(np.diff(angulos)))

            score = (
                exceso_pendiente * 100000
                + porcentaje_fuera * 50000
                + falta_longitud * 5
                + suavidad * 150
                + ondas * 10
            )

            if score < mejor_score:
                mejor_score = score
                mejor_eje = eje_xy

    longitud_total = calcular_distancias_acumuladas(mejor_eje)[-1]

    estaciones = np.arange(0, longitud_total, 10)

    if len(estaciones) == 0 or estaciones[-1] < longitud_total:
        estaciones = np.append(estaciones, longitud_total)

    eje_suavizado = interpolar_eje(mejor_eje, estaciones)

    return {
        "xy": eje_suavizado,
        "inicio": inicio,
        "final": final,
        "longitud_total": longitud_total,
        "desnivel": desnivel
    }


def construir_rasante_vial(df, eje_xy, pendientes_tramos):
    interp_lineal, interp_cercano = crear_interpoladores_terreno(df)

    distancias = calcular_distancias_acumuladas(eje_xy)
    z_terreno = obtener_z_terreno(eje_xy, interp_lineal, interp_cercano)

    z_min = df["Z"].min()
    z_max = df["Z"].max()
    relieve = z_max - z_min

    separacion_visual = max(relieve * 0.003, 0.10)
    profundidad_corte_objetivo = max(relieve * 0.075, 2.80)

    z_inicio_objetivo = z_terreno[0] + separacion_visual
    z_final_objetivo = z_terreno[-1] + separacion_visual

    z_rasante = [z_inicio_objetivo]

    for i in range(1, len(distancias)):
        estacion_anterior = distancias[i - 1]
        estacion_actual = distancias[i]
        delta = estacion_actual - estacion_anterior

        tramo = int(estacion_anterior // 100)
        pendiente = pendientes_tramos.get(tramo, 0) / 100

        z_rasante.append(z_rasante[-1] + pendiente * delta)

    z_rasante = np.array(z_rasante)

    error_final = z_final_objetivo - z_rasante[-1]
    correccion = np.linspace(0, error_final, len(z_rasante))
    z_rasante = z_rasante + correccion

    t = np.linspace(0, 1, len(z_rasante))
    perfil_corte = np.sin(np.pi * t) ** 0.85
    z_rasante = z_rasante - profundidad_corte_objetivo * perfil_corte

    error_final = z_final_objetivo - z_rasante[-1]
    correccion = np.linspace(0, error_final, len(z_rasante))
    z_rasante = z_rasante + correccion

    limite_corte = max(relieve * 0.22, 5.0)
    limite_relleno = max(relieve * 0.008, 0.25)

    diferencia = z_rasante - z_terreno
    diferencia = np.clip(diferencia, -limite_corte, limite_relleno)

    z_rasante = z_terreno + diferencia

    z_rasante[0] = z_inicio_objetivo
    z_rasante[-1] = z_final_objetivo

    return {
        "xy": eje_xy,
        "estaciones": distancias,
        "z_terreno": z_terreno,
        "z_rasante": z_rasante,
        "longitud_total": distancias[-1]
    }



def interpolar_rasante_en_estacion(rasante, estacion_objetivo):
    estaciones = rasante["estaciones"]
    xy = rasante["xy"]

    x = np.interp(estacion_objetivo, estaciones, xy[:, 0])
    y = np.interp(estacion_objetivo, estaciones, xy[:, 1])
    z_terreno = np.interp(estacion_objetivo, estaciones, rasante["z_terreno"])
    z_rasante = np.interp(estacion_objetivo, estaciones, rasante["z_rasante"])

    return np.array([x, y]), z_terreno, z_rasante


def calcular_volumen_acumulado_rasante(rasante, ancho_via):
    estaciones = rasante["estaciones"]
    z_terreno = rasante["z_terreno"]
    z_rasante = rasante["z_rasante"]

    volumenes = [0.0]
    cortes = [0.0]
    rellenos = [0.0]

    for i in range(len(estaciones) - 1):
        longitud_tramo = estaciones[i + 1] - estaciones[i]
        dif_1 = z_rasante[i] - z_terreno[i]
        dif_2 = z_rasante[i + 1] - z_terreno[i + 1]
        diferencia_media = (dif_1 + dif_2) / 2
        volumen = abs(diferencia_media) * ancho_via * longitud_tramo

        volumenes.append(volumenes[-1] + volumen)

        if diferencia_media < 0:
            cortes.append(cortes[-1] + volumen)
            rellenos.append(rellenos[-1])
        else:
            cortes.append(cortes[-1])
            rellenos.append(rellenos[-1] + volumen)

    return np.array(volumenes), np.array(cortes), np.array(rellenos)


def recortar_rasante_por_presupuesto(rasante, ancho_via, presupuesto_tierra):
    presupuesto_tierra = float(presupuesto_tierra)
    volumen_total, corte_total, relleno_total = calcular_volumen_acumulado_rasante(
        rasante,
        ancho_via
    )

    if presupuesto_tierra <= 0:
        presupuesto_tierra = 1.0

    if volumen_total[-1] <= presupuesto_tierra:
        rasante["longitud_total_diseno"] = rasante["longitud_total"]
        rasante["presupuesto_usado"] = volumen_total[-1]
        rasante["volumen_corte_estimado"] = corte_total[-1]
        rasante["volumen_relleno_estimado"] = relleno_total[-1]
        rasante["presupuesto_suficiente"] = True
        return rasante

    idx = int(np.searchsorted(volumen_total, presupuesto_tierra, side="right"))
    idx = max(1, min(idx, len(volumen_total) - 1))

    volumen_anterior = volumen_total[idx - 1]
    volumen_segmento = volumen_total[idx] - volumen_anterior
    if volumen_segmento <= 0:
        fraccion = 0.0
    else:
        fraccion = (presupuesto_tierra - volumen_anterior) / volumen_segmento
        fraccion = float(np.clip(fraccion, 0.0, 1.0))

    estacion_inicio = rasante["estaciones"][idx - 1]
    estacion_fin = rasante["estaciones"][idx]
    estacion_corte = estacion_inicio + fraccion * (estacion_fin - estacion_inicio)

    xy_corte, z_terreno_corte, z_rasante_corte = interpolar_rasante_en_estacion(
        rasante,
        estacion_corte
    )

    incluir = rasante["estaciones"] < estacion_corte

    estaciones_nuevas = list(rasante["estaciones"][incluir])
    xy_nuevas = list(rasante["xy"][incluir])
    z_terreno_nuevo = list(rasante["z_terreno"][incluir])
    z_rasante_nuevo = list(rasante["z_rasante"][incluir])

    if len(estaciones_nuevas) == 0:
        estaciones_nuevas = [0.0]
        xy_nuevas = [rasante["xy"][0]]
        z_terreno_nuevo = [rasante["z_terreno"][0]]
        z_rasante_nuevo = [rasante["z_rasante"][0]]

    if estacion_corte > estaciones_nuevas[-1]:
        estaciones_nuevas.append(estacion_corte)
        xy_nuevas.append(xy_corte)
        z_terreno_nuevo.append(z_terreno_corte)
        z_rasante_nuevo.append(z_rasante_corte)

    rasante_recortada = {
        "xy": np.array(xy_nuevas),
        "estaciones": np.array(estaciones_nuevas),
        "z_terreno": np.array(z_terreno_nuevo),
        "z_rasante": np.array(z_rasante_nuevo),
        "longitud_total": float(estacion_corte),
        "longitud_total_diseno": float(rasante["longitud_total"]),
        "presupuesto_usado": float(presupuesto_tierra),
        "volumen_corte_estimado": float(np.interp(estacion_corte, rasante["estaciones"], corte_total)),
        "volumen_relleno_estimado": float(np.interp(estacion_corte, rasante["estaciones"], relleno_total)),
        "presupuesto_suficiente": False
    }

    return rasante_recortada


def calcular_bordes_via(rasante, ancho_via):
    xy = rasante["xy"]

    tangentes = np.zeros_like(xy)
    tangentes[1:-1] = xy[2:] - xy[:-2]
    tangentes[0] = xy[1] - xy[0]
    tangentes[-1] = xy[-1] - xy[-2]

    normales = np.column_stack([-tangentes[:, 1], tangentes[:, 0]])
    normas = np.linalg.norm(normales, axis=1)
    normas[normas == 0] = 1
    normales = normales / normas[:, None]

    borde_izquierdo = xy + normales * (ancho_via / 2)
    borde_derecho = xy - normales * (ancho_via / 2)

    return borde_izquierdo, borde_derecho


def agregar_franja_via(fig, rasante, ancho_via):
    borde_izq, borde_der = calcular_bordes_via(rasante, ancho_via)

    z = rasante["z_rasante"] + 0.08

    x = list(borde_izq[:, 0]) + list(borde_der[:, 0])
    y = list(borde_izq[:, 1]) + list(borde_der[:, 1])
    z_vertices = list(z) + list(z)

    n = len(rasante["xy"])
    i, j, k = [], [], []

    for a in range(n - 1):
        i += [a, a + 1]
        j += [a + n, a + n]
        k += [a + 1, a + 1 + n]

    fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=z_vertices,
        i=i,
        j=j,
        k=k,
        color="rgb(55, 55, 55)",
        opacity=0.65,
        flatshading=True,
        name="Ancho de vía",
        hoverinfo="skip"
    ))


def agregar_tramos_coloreados_rasante(fig, rasante):
    colores = ["red", "lime", "cyan", "magenta", "orange", "yellow"]

    estaciones = rasante["estaciones"]
    xy = rasante["xy"]
    z = rasante["z_rasante"] + 0.18

    cantidad_tramos = int(np.ceil(rasante["longitud_total"] / 100))

    for tramo in range(cantidad_tramos):
        inicio = tramo * 100
        fin = min((tramo + 1) * 100, rasante["longitud_total"])

        mascara = (estaciones >= inicio) & (estaciones <= fin)

        if np.sum(mascara) >= 2:
            fig.add_trace(go.Scatter3d(
                x=xy[mascara, 0],
                y=xy[mascara, 1],
                z=z[mascara],
                mode="lines",
                line=dict(color=colores[tramo % len(colores)], width=9),
                name=f"Tramo {formato_km(inicio)}"
            ))


def agregar_lineas_verticales_rasante(fig, rasante, estaciones):
    xy_estacas = interpolar_eje(rasante["xy"], estaciones)

    z_terreno = np.interp(estaciones, rasante["estaciones"], rasante["z_terreno"])
    z_rasante = np.interp(estaciones, rasante["estaciones"], rasante["z_rasante"])

    for x, y, zt, zr in zip(xy_estacas[:, 0], xy_estacas[:, 1], z_terreno, z_rasante):
        partes = 18
        zs = np.linspace(zt, zr, partes)

        xs, ys, z_plot = [], [], []

        for i in range(0, partes - 1, 2):
            xs += [x, x, None]
            ys += [y, y, None]
            z_plot += [zs[i], zs[i + 1], None]

        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=z_plot,
            mode="lines",
            line=dict(color="yellow", width=3),
            showlegend=False,
            hoverinfo="skip"
        ))


def agregar_corte_relleno_3d(fig, rasante, ancho_via):
    xy = rasante["xy"]
    estaciones = rasante["estaciones"]
    z_terreno = rasante["z_terreno"]
    z_rasante = rasante["z_rasante"]

    borde_izq, borde_der = calcular_bordes_via(rasante, ancho_via)

    volumen_corte = 0
    volumen_relleno = 0

    corte_agregado = False
    relleno_agregado = False

    for i in range(len(xy) - 1):
        longitud_tramo = estaciones[i + 1] - estaciones[i]

        dif_1 = z_rasante[i] - z_terreno[i]
        dif_2 = z_rasante[i + 1] - z_terreno[i + 1]
        diferencia_media = (dif_1 + dif_2) / 2

        volumen = abs(diferencia_media) * ancho_via * longitud_tramo

        if diferencia_media < 0:
            volumen_corte += volumen
            color = "rgba(220, 60, 35, 0.60)"
            nombre = "Corte"
            mostrar_leyenda = not corte_agregado
            corte_agregado = True
        else:
            volumen_relleno += volumen
            color = "rgba(35, 170, 90, 0.60)"
            nombre = "Relleno"
            mostrar_leyenda = not relleno_agregado
            relleno_agregado = True

        x = [
            borde_izq[i, 0], borde_der[i, 0],
            borde_izq[i + 1, 0], borde_der[i + 1, 0],
            borde_izq[i, 0], borde_der[i, 0],
            borde_izq[i + 1, 0], borde_der[i + 1, 0]
        ]

        y = [
            borde_izq[i, 1], borde_der[i, 1],
            borde_izq[i + 1, 1], borde_der[i + 1, 1],
            borde_izq[i, 1], borde_der[i, 1],
            borde_izq[i + 1, 1], borde_der[i + 1, 1]
        ]

        z = [
            z_terreno[i], z_terreno[i],
            z_terreno[i + 1], z_terreno[i + 1],
            z_rasante[i], z_rasante[i],
            z_rasante[i + 1], z_rasante[i + 1]
        ]

        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=[0, 0, 4, 4, 0, 1, 2, 3, 0, 2, 1, 3],
            j=[1, 2, 5, 6, 4, 5, 6, 7, 1, 3, 5, 7],
            k=[3, 3, 7, 7, 5, 4, 7, 6, 5, 7, 7, 5],
            color=color,
            opacity=0.60,
            flatshading=True,
            name=nombre,
            showlegend=mostrar_leyenda,
            hovertemplate=(
                f"{nombre}<br>"
                f"Volumen aprox.: {volumen:.2f} m³"
                "<extra></extra>"
            )
        ))

    return volumen_corte, volumen_relleno


def agregar_zanja_excavacion(fig, rasante, ancho_via):
    xy = rasante["xy"]
    z_terreno = rasante["z_terreno"]
    z_rasante = rasante["z_rasante"]

    borde_izq, borde_der = calcular_bordes_via(rasante, ancho_via)

    z_fondo = z_rasante + 0.05
    corte = z_rasante < z_terreno

    zanja_agregada = False

    for i in range(len(xy) - 1):
        if not (corte[i] or corte[i + 1]):
            continue

        profundidad_1 = max(z_terreno[i] - z_rasante[i], 0)
        profundidad_2 = max(z_terreno[i + 1] - z_rasante[i + 1], 0)

        if profundidad_1 <= 0 and profundidad_2 <= 0:
            continue

        x = [
            borde_izq[i, 0], borde_der[i, 0],
            borde_izq[i + 1, 0], borde_der[i + 1, 0],
            borde_izq[i, 0], borde_der[i, 0],
            borde_izq[i + 1, 0], borde_der[i + 1, 0],
        ]

        y = [
            borde_izq[i, 1], borde_der[i, 1],
            borde_izq[i + 1, 1], borde_der[i + 1, 1],
            borde_izq[i, 1], borde_der[i, 1],
            borde_izq[i + 1, 1], borde_der[i + 1, 1],
        ]

        z = [
            z_terreno[i], z_terreno[i],
            z_terreno[i + 1], z_terreno[i + 1],
            z_fondo[i], z_fondo[i],
            z_fondo[i + 1], z_fondo[i + 1],
        ]

        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=[0, 0, 4, 4, 0, 1, 2, 3, 0, 2, 1, 3],
            j=[1, 2, 5, 6, 4, 5, 6, 7, 1, 3, 5, 7],
            k=[3, 3, 7, 7, 5, 4, 7, 6, 5, 7, 7, 5],
            color="rgba(150, 82, 28, 0.92)",
            opacity=0.92,
            flatshading=True,
            name="Zanja de corte",
            showlegend=not zanja_agregada,
            hovertemplate=(
                "Excavación / Corte<br>"
                f"Profundidad inicial: {profundidad_1:.2f} m<br>"
                f"Profundidad final: {profundidad_2:.2f} m"
                "<extra></extra>"
            )
        ))

        zanja_agregada = True

        fig.add_trace(go.Scatter3d(
            x=[
                borde_izq[i, 0],
                borde_izq[i + 1, 0],
                None,
                borde_der[i, 0],
                borde_der[i + 1, 0]
            ],
            y=[
                borde_izq[i, 1],
                borde_izq[i + 1, 1],
                None,
                borde_der[i, 1],
                borde_der[i + 1, 1]
            ],
            z=[
                z_terreno[i] + 0.05,
                z_terreno[i + 1] + 0.05,
                None,
                z_terreno[i] + 0.05,
                z_terreno[i + 1] + 0.05
            ],
            mode="lines",
            line=dict(color="orange", width=6),
            showlegend=False,
            hoverinfo="skip"
        ))


def crear_figura_maqueta(df, incluir_estratos=True):
    puntos = df[["X", "Y", "Z"]].values
    puntos_2d = df[["X", "Y"]].values

    z_min = df["Z"].min()
    z_max = df["Z"].max()
    relieve = z_max - z_min

    cota_base = z_min - max(relieve * 0.55, 10)

    tri = Delaunay(puntos_2d)
    hull = ConvexHull(puntos_2d)
    perimetro = hull.vertices

    centro = puntos_2d[perimetro].mean(axis=0)
    angulos = np.arctan2(
        puntos_2d[perimetro, 1] - centro[1],
        puntos_2d[perimetro, 0] - centro[0]
    )
    perimetro = perimetro[np.argsort(angulos)]

    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=puntos[:, 0],
        y=puntos[:, 1],
        z=puntos[:, 2],
        i=tri.simplices[:, 0],
        j=tri.simplices[:, 1],
        k=tri.simplices[:, 2],
        intensity=puntos[:, 2],
        colorscale=[
            [0.00, "rgb(142, 76, 25)"],
            [0.35, "rgb(199, 127, 46)"],
            [0.65, "rgb(215, 194, 111)"],
            [1.00, "rgb(90, 145, 75)"]
        ],
        colorbar=dict(title="Elevación (m)", orientation="v"),
        opacity=1,
        flatshading=False,
        lighting=dict(
            ambient=0.35,
            diffuse=0.9,
            specular=0.25,
            roughness=0.5,
            fresnel=0.15
        ),
        lightposition=dict(x=100, y=200, z=500),
        hovertemplate="X: %{x}<br>Y: %{y}<br>Elevación: %{z:.2f} m<extra></extra>",
        name="Superficie"
    ))

    x_pared, y_pared, z_pared = [], [], []

    for idx in perimetro:
        x_pared.append(puntos[idx, 0])
        y_pared.append(puntos[idx, 1])
        z_pared.append(puntos[idx, 2])

    for idx in perimetro:
        x_pared.append(puntos[idx, 0])
        y_pared.append(puntos[idx, 1])
        z_pared.append(cota_base)

    n = len(perimetro)
    i_pared, j_pared, k_pared = [], [], []

    for a in range(n):
        b = (a + 1) % n

        arriba_a = a
        arriba_b = b
        abajo_a = a + n
        abajo_b = b + n

        i_pared += [arriba_a, arriba_a]
        j_pared += [arriba_b, abajo_b]
        k_pared += [abajo_b, abajo_a]

    fig.add_trace(go.Mesh3d(
        x=x_pared,
        y=y_pared,
        z=z_pared,
        i=i_pared,
        j=j_pared,
        k=k_pared,
        intensity=z_pared,
        colorscale=[
            [0.00, "rgb(42, 23, 10)"],
            [0.50, "rgb(118, 62, 23)"],
            [1.00, "rgb(190, 102, 35)"]
        ],
        showscale=False,
        opacity=1,
        flatshading=True,
        lighting=dict(
            ambient=0.22,
            diffuse=0.85,
            specular=0.12,
            roughness=0.9
        ),
        lightposition=dict(x=100, y=200, z=500),
        hoverinfo="skip",
        name="Paredes"
    ))

    x_base = list(puntos[perimetro, 0])
    y_base = list(puntos[perimetro, 1])
    z_base = [cota_base] * n

    x_base.append(np.mean(x_base))
    y_base.append(np.mean(y_base))
    z_base.append(cota_base)

    centro_base = n
    i_base, j_base, k_base = [], [], []

    for a in range(n):
        b = (a + 1) % n
        i_base.append(centro_base)
        j_base.append(a)
        k_base.append(b)

    fig.add_trace(go.Mesh3d(
        x=x_base,
        y=y_base,
        z=z_base,
        i=i_base,
        j=j_base,
        k=k_base,
        color="rgb(35, 20, 10)",
        opacity=1,
        flatshading=True,
        hoverinfo="skip",
        name="Base"
    ))

    agregar_curvas_superficie(
        fig,
        puntos,
        tri,
        z_min,
        z_max,
        color="rgb(35, 55, 65)",
        ancho=4
    )

    if incluir_estratos:
        intervalo = 5

        niveles_pared = np.arange(
            np.floor(cota_base / intervalo) * intervalo,
            np.ceil(z_max / intervalo) * intervalo + intervalo,
            intervalo
        )

        rango_xy = max(
            puntos[:, 0].max() - puntos[:, 0].min(),
            puntos[:, 1].max() - puntos[:, 1].min()
        )

        separacion_visual = rango_xy * 0.0018

        for nivel in niveles_pared:
            xs, ys, zs = [], [], []

            for a in range(n):
                b = (a + 1) % n

                idx1 = perimetro[a]
                idx2 = perimetro[b]

                p1 = puntos[idx1]
                p2 = puntos[idx2]

                vertices_pared = [
                    np.array([p1[0], p1[1], p1[2]]),
                    np.array([p2[0], p2[1], p2[2]]),
                    np.array([p2[0], p2[1], cota_base]),
                    np.array([p1[0], p1[1], cota_base])
                ]

                aristas = [(0, 1), (1, 2), (2, 3), (3, 0)]
                cortes = []

                for e1, e2 in aristas:
                    q1 = vertices_pared[e1]
                    q2 = vertices_pared[e2]
                    z1 = q1[2]
                    z2 = q2[2]

                    if (z1 <= nivel <= z2) or (z2 <= nivel <= z1):
                        if z1 != z2:
                            t = (nivel - z1) / (z2 - z1)
                            x = q1[0] + t * (q2[0] - q1[0])
                            y = q1[1] + t * (q2[1] - q1[1])
                            z = nivel
                            cortes.append([x, y, z])

                cortes_unicos = []

                for c in cortes:
                    if not any(np.linalg.norm(np.array(c) - np.array(u)) < 0.001 for u in cortes_unicos):
                        cortes_unicos.append(c)

                if len(cortes_unicos) >= 2:
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2

                    direccion = np.array([mid_x - centro[0], mid_y - centro[1]])
                    norma = np.linalg.norm(direccion)

                    if norma > 0:
                        direccion = direccion / norma
                    else:
                        direccion = np.array([0, 0])

                    dx = direccion[0] * separacion_visual
                    dy = direccion[1] * separacion_visual

                    xs += [cortes_unicos[0][0] + dx, cortes_unicos[1][0] + dx, None]
                    ys += [cortes_unicos[0][1] + dy, cortes_unicos[1][1] + dy, None]
                    zs += [cortes_unicos[0][2], cortes_unicos[1][2], None]

            if xs:
                fig.add_trace(go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color="rgb(45, 68, 78)", width=5),
                    showlegend=False,
                    hoverinfo="skip"
                ))

    fig.update_layout(
        height=820,
        template="plotly_dark",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            bgcolor="rgb(5,5,5)",
            aspectmode="manual",
            aspectratio=dict(x=1.6, y=1.0, z=0.55),
            camera=dict(eye=dict(x=1.7, y=1.7, z=0.95))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )

    return fig


def texto_pdf(txt):
    return str(txt).replace("³", "3")


def generar_memoria_pdf(nombre_proyecto, resumen, simulaciones, capturas):
    if FPDF is None:
        raise ModuleNotFoundError("Instala fpdf2 con: pip install fpdf2")

    class MemoriaPDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_fill_color(36, 62, 82)
            self.rect(0, 0, 210, 12, "F")
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 9)
            self.cell(0, 8, texto_pdf("Memoria de Calculo - Diseño Vial y Movimiento de Tierras"), ln=True, align="C")
            self.set_text_color(0, 0, 0)

        def footer(self):
            self.set_y(-15)
            self.set_draw_color(180, 180, 180)
            self.line(15, self.get_y(), 195, self.get_y())
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(90, 90, 90)
            self.cell(0, 8, texto_pdf(f"Pagina {self.page_no()}"), align="C")
            self.set_text_color(0, 0, 0)

    def titulo_seccion(pdf, titulo):
        pdf.ln(4)
        pdf.set_fill_color(230, 236, 242)
        pdf.set_text_color(36, 62, 82)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, texto_pdf(titulo), ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    def fila_tabla(pdf, izquierda, derecha, fill=False):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_fill_color(247, 249, 251 if fill else 255)
        pdf.cell(70, 8, texto_pdf(izquierda), border=1, fill=fill)
        pdf.cell(70, 8, texto_pdf(derecha), border=1, ln=True, fill=fill)

    pdf = MemoriaPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    fecha = datetime.now().strftime("%Y-%m-%d")

    # Portada
    pdf.add_page()
    pdf.set_fill_color(36, 62, 82)
    pdf.rect(0, 0, 210, 55, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.ln(14)
    pdf.cell(0, 10, texto_pdf("Universidad / Carrera"), ln=True, align="C")
    pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(
        0,
        11,
        texto_pdf("Memoria de Calculo - Diseño Vial y Movimiento de Tierras"),
        align="C"
    )

    pdf.set_text_color(40, 40, 40)
    pdf.ln(28)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, texto_pdf(f"Proyecto: {nombre_proyecto}"), ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, texto_pdf(f"Fecha de emision: {fecha}"), ln=True, align="C")
    pdf.ln(20)
    pdf.set_draw_color(36, 62, 82)
    pdf.set_line_width(0.7)
    pdf.line(35, pdf.get_y(), 175, pdf.get_y())
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        7,
        texto_pdf("Documento tecnico generado a partir de los parametros, calculos volumetricos y simulaciones guardadas en la aplicacion."),
        align="C"
    )

    # Resumen
    pdf.add_page()
    titulo_seccion(pdf, "1. Resumen Ejecutivo del Proyecto")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        7,
        texto_pdf("Este resumen presenta los parametros de diseño y resultados volumetricos vigentes al momento de generar la memoria."),
    )
    pdf.ln(3)

    datos = [
        ("Ancho de via", f"{resumen['ancho']:.2f} m"),
        ("Presupuesto de tierra", f"{resumen['presupuesto']:.2f} m3"),
        ("Longitud construida", f"{resumen['longitud']:.2f} m"),
        ("Volumen de corte", f"{resumen['corte']:.2f} m3"),
        ("Volumen de relleno", f"{resumen['relleno']:.2f} m3"),
    ]

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(36, 62, 82)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(70, 8, texto_pdf("Parametro"), border=1, fill=True)
    pdf.cell(70, 8, texto_pdf("Valor"), border=1, ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)

    for idx, (etiqueta, valor) in enumerate(datos):
        fila_tabla(pdf, etiqueta, valor, fill=idx % 2 == 0)

    # Tabla comparativa
    titulo_seccion(pdf, "2. Tabla Comparativa de Simulaciones")

    if simulaciones.empty:
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 8, texto_pdf("No existen simulaciones guardadas en la base de datos."))
    else:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(36, 62, 82)
        pdf.set_text_color(255, 255, 255)
        columnas = ["nombre", "ancho", "presupuesto", "longitud_lograda", "corte", "relleno"]
        anchos = [42, 22, 28, 32, 30, 30]

        for col, ancho in zip(columnas, anchos):
            pdf.cell(ancho, 7, texto_pdf(col), border=1, fill=True)
        pdf.ln()

        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 8)

        for row_idx, (_, fila) in enumerate(simulaciones.head(12).iterrows()):
            pdf.set_fill_color(247, 249, 251 if row_idx % 2 == 0 else 255)
            valores = [
                str(fila["nombre"])[:22],
                f"{fila['ancho']:.2f}",
                f"{fila['presupuesto']:.2f}",
                f"{fila['longitud_lograda']:.2f}",
                f"{fila['corte']:.2f}",
                f"{fila['relleno']:.2f}",
            ]

            for valor, ancho in zip(valores, anchos):
                pdf.cell(ancho, 7, texto_pdf(valor), border=1, fill=True)
            pdf.ln()

    # Capturas
    titulos = [
        "Figura 1. Modelo 3D del terreno",
        "Figura 2. Trazado vial",
        "Figura 3. Maqueta de excavacion"
    ]

    archivos_temporales = []

    for i, captura in enumerate(capturas):
        suffix = os.path.splitext(captura.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(captura.getvalue())
            ruta_tmp = tmp.name
            archivos_temporales.append(ruta_tmp)

        pdf.add_page()
        titulo = titulos[i] if i < len(titulos) else f"Figura {i + 1}. Captura del modelo 3D"
        titulo_seccion(pdf, titulo)

        try:
            pdf.image(ruta_tmp, x=15, y=32, w=180)
        except Exception:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 8, texto_pdf("No se pudo insertar esta imagen en el PDF."))

    # Conclusión
    pdf.add_page()
    titulo_seccion(pdf, "3. Conclusion Automatica")
    pdf.set_font("Helvetica", "", 11)

    if resumen["corte"] > resumen["presupuesto"]:
        conclusion = (
            "El volumen de corte supera el presupuesto de tierra establecido. "
            "Por lo tanto, se requiere optimizar el trazado vial, revisar pendientes "
            "o modificar la geometria para reducir el movimiento de tierras."
        )
    else:
        conclusion = (
            "El volumen de corte se encuentra dentro del presupuesto de tierra establecido. "
            "La alternativa evaluada es viable de manera preliminar, sujeta a revision tecnica "
            "detallada y validacion en campo."
        )

    pdf.set_fill_color(247, 249, 251)
    pdf.multi_cell(0, 8, texto_pdf(conclusion), border=1, fill=True)

    salida = pdf.output(dest="S")

    for archivo in archivos_temporales:
        try:
            os.remove(archivo)
        except Exception:
            pass

    if isinstance(salida, bytearray):
        return bytes(salida)

    return salida.encode("latin-1")


# ==================================================
# CARGA DE ARCHIVO ORIGINAL
# ==================================================
archivo = st.file_uploader("Suba un archivo TXT o CSV", type=["txt", "csv"])

if archivo is not None:
    try:
        df = pd.read_csv(archivo, names=["ID", "X", "Y", "Z", "Etiqueta"])
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df["Z"] = pd.to_numeric(df["Z"], errors="coerce")
        df = df.dropna()

        st.success(f"✅ Se han cargado exitosamente {len(df)} puntos topográficos.")

        if opcion == "Estadísticas":
            st.subheader("📊 Resumen del Levantamiento")

            col1, col2, col3 = st.columns(3)
            col1.metric("Cota Máxima", f"{df['Z'].max():.2f} m")
            col2.metric("Cota Mínima", f"{df['Z'].min():.2f} m")
            col3.metric("Cantidad de Puntos", len(df))

            st.dataframe(df.head(20))

        elif opcion == "Nube de Puntos 3D":
            st.subheader("☁️ Nube de Puntos Topográficos")

            fig = px.scatter_3d(
                df,
                x="X",
                y="Y",
                z="Z",
                color="Z",
                color_continuous_scale="earth"
            )

            fig.update_traces(marker=dict(size=3))
            fig.update_layout(height=700)

            st.plotly_chart(fig, use_container_width=True)

        elif opcion == "Esqueleto Estructural (TIN)":
            st.subheader("🕸️ Esqueleto Estructural (TIN)")

            max_arista = st.sidebar.slider(
                "Longitud máxima de arista (m):",
                min_value=1.0,
                max_value=500.0,
                value=68.3,
                step=0.1
            )

            if st.sidebar.button("Generar/Actualizar TIN"):
                puntos = df[["X", "Y", "Z"]].values
                puntos_2d = df[["X", "Y"]].values
                tri = Delaunay(puntos_2d)

                edges = set()

                for s in tri.simplices:
                    p1, p2, p3 = puntos_2d[s]

                    if (
                        np.linalg.norm(p1 - p2) <= max_arista
                        and np.linalg.norm(p2 - p3) <= max_arista
                        and np.linalg.norm(p3 - p1) <= max_arista
                    ):
                        for a, b in [(s[0], s[1]), (s[1], s[2]), (s[2], s[0])]:
                            edges.add(tuple(sorted((a, b))))

                fig_tin = go.Figure()

                for i, j in edges:
                    fig_tin.add_trace(go.Scatter3d(
                        x=[puntos[i, 0], puntos[j, 0]],
                        y=[puntos[i, 1], puntos[j, 1]],
                        z=[puntos[i, 2], puntos[j, 2]],
                        mode="lines",
                        line=dict(color="red", width=2),
                        showlegend=False
                    ))

                fig_tin.update_layout(
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=0)
                )

                st.session_state.tin_fig = fig_tin

            if st.session_state.tin_fig:
                st.plotly_chart(st.session_state.tin_fig, use_container_width=True)
            else:
                st.info("Presione 'Generar/Actualizar TIN' en el menú lateral para visualizar.")

        elif opcion == "Superficie Sólida (MDE)":
            st.subheader("🏔️ Superficie Sólida (MDE)")
            st.info("Curvas de nivel fijadas cada 5 metros. Pasa el cursor sobre la montaña para leer la elevación.")

            puntos = df[["X", "Y", "Z"]].values
            puntos_2d = df[["X", "Y"]].values
            tri = Delaunay(puntos_2d)

            z_min = df["Z"].min()
            z_max = df["Z"].max()

            fig_mde = go.Figure()

            fig_mde.add_trace(go.Mesh3d(
                x=puntos[:, 0],
                y=puntos[:, 1],
                z=puntos[:, 2],
                i=tri.simplices[:, 0],
                j=tri.simplices[:, 1],
                k=tri.simplices[:, 2],
                intensity=puntos[:, 2],
                colorscale="Earth",
                colorbar=dict(title="Elevación (m)", orientation="v"),
                opacity=1,
                lighting=dict(
                    ambient=0.35,
                    diffuse=0.85,
                    specular=0.25,
                    roughness=0.55,
                    fresnel=0.15
                ),
                lightposition=dict(x=100, y=200, z=400),
                hovertemplate="X: %{x}<br>Y: %{y}<br>Elevación: %{z:.2f} m<extra></extra>"
            ))

            agregar_curvas_superficie(
                fig_mde,
                puntos,
                tri,
                z_min,
                z_max,
                color="white",
                ancho=6
            )

            fig_mde.update_layout(
                height=750,
                template="plotly_dark",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Elevación Z",
                    bgcolor="rgb(8,8,8)",
                    aspectmode="manual",
                    aspectratio=dict(x=1.6, y=1.0, z=0.45),
                    camera=dict(eye=dict(x=1.7, y=1.7, z=1.0))
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )

            st.plotly_chart(fig_mde, use_container_width=True)

        elif opcion == "Maqueta Topográfica (Bloque 3D)":
            st.subheader("🏗️ Maqueta Topográfica (Bloque 3D)")
            st.info("Maqueta física del terreno: bloque sólido con paredes verticales, base y estratos visibles cada 5 metros.")

            fig_bloque = crear_figura_maqueta(df, incluir_estratos=True)
            st.plotly_chart(fig_bloque, use_container_width=True)

        elif opcion == "Parámetros de Diseño":
            st.title("🚜 Simulador Vial y Movimiento de Tierras")
            st.subheader("Fase 6: Geometría y Presupuesto")

            parametros_actuales = st.session_state.parametros_diseno or {
                "ancho_via": 10.0,
                "presupuesto_tierra": 40000.0
            }

            col1, col2 = st.columns(2)

            with col1:
                ancho_via = st.number_input(
                    "Ancho de vía (W en metros)",
                    min_value=0.1,
                    value=float(parametros_actuales.get("ancho_via", 10.0)),
                    step=0.5
                )

            with col2:
                presupuesto_tierra = st.number_input(
                    "Presupuesto de Tierra (Volumen máximo m³)",
                    min_value=1.0,
                    value=float(parametros_actuales.get("presupuesto_tierra", 40000.0)),
                    step=100.0
                )

            if st.button("Guardar/Actualizar Parámetros", use_container_width=True):
                if ancho_via <= 0 or presupuesto_tierra <= 0:
                    st.error("Ingrese valores mayores que cero.")
                else:
                    nuevos_parametros = {
                        "ancho_via": float(ancho_via),
                        "presupuesto_tierra": float(presupuesto_tierra)
                    }

                    parametros_cambiaron = nuevos_parametros != st.session_state.parametros_diseno
                    st.session_state.parametros_diseno = nuevos_parametros

                    if parametros_cambiaron:
                        st.session_state.rasante = None
                        st.session_state.excavacion = None
                        st.session_state.pdf_memoria = None

                    st.success("Parámetros actualizados. Recalcula la Fase 7 para aplicar los cambios.")

            if st.session_state.parametros_diseno is not None:
                parametros = st.session_state.parametros_diseno
                col1, col2 = st.columns(2)
                col1.metric("Ancho de vía guardado", f"{parametros['ancho_via']:.2f} m")
                col2.metric("Presupuesto de tierra guardado", f"{parametros['presupuesto_tierra']:.2f} m³")

        elif opcion == "Diseño de Eje y Rasante":
            st.subheader("🛣️ Diseño de Eje y Rasante")

            if st.session_state.parametros_diseno is None:
                st.warning("Primero debe completar la Fase 6: Parámetros de Diseño.")

            else:
                parametros = st.session_state.parametros_diseno
                ancho_via = parametros["ancho_via"]

                if st.session_state.eje_vial is None:
                    st.session_state.eje_vial = generar_eje_vial_automatico(df)
                    st.session_state.longitud_total = st.session_state.eje_vial["longitud_total"]

                col_a, col_b = st.columns([3, 1])

                with col_a:
                    st.info(
                        f"Longitud total trazada: "
                        f"{st.session_state.eje_vial['longitud_total']:.2f} metros."
                    )

                with col_b:
                    if st.button("Regenerar eje vial", use_container_width=True):
                        st.session_state.eje_vial = generar_eje_vial_automatico(df)
                        st.session_state.longitud_total = st.session_state.eje_vial["longitud_total"]
                        st.session_state.rasante = None
                        st.session_state.pendientes_tramos = {}
                        st.session_state.excavacion = None
                        st.rerun()

                eje_vial = st.session_state.eje_vial
                longitud_total = eje_vial["longitud_total"]

                cantidad_tramos = int(np.ceil(longitud_total / 100))

                pendiente_recomendada = abs(eje_vial["desnivel"]) / longitud_total * 100
                pendiente_recomendada = min(max(pendiente_recomendada, 3), 10)

                st.caption(f"Pendiente recomendada automática: {pendiente_recomendada:.2f}%")

                columnas = st.columns(4)
                pendientes_tramos = {}

                for tramo in range(cantidad_tramos):
                    inicio = tramo * 100
                    fin = min((tramo + 1) * 100, longitud_total)

                    etiqueta = f"{formato_km(inicio)} a {formato_km(fin)} (%)"

                    valor_guardado = st.session_state.pendientes_tramos.get(
                        tramo,
                        pendiente_recomendada
                    )

                    with columnas[tramo % 4]:
                        pendiente = st.number_input(
                            etiqueta,
                            value=float(valor_guardado),
                            step=0.5,
                            key=f"pendiente_tramo_{tramo}"
                        )

                        pendientes_tramos[tramo] = pendiente

                        if abs(pendiente) > 12:
                            st.warning("Pendiente alta")

                if st.button("Calcular Rasante Multitramo en 3D", use_container_width=True):
                    rasante_completa = construir_rasante_vial(
                        df=df,
                        eje_xy=eje_vial["xy"],
                        pendientes_tramos=pendientes_tramos
                    )

                    rasante = recortar_rasante_por_presupuesto(
                        rasante=rasante_completa,
                        ancho_via=ancho_via,
                        presupuesto_tierra=parametros["presupuesto_tierra"]
                    )

                    st.session_state.eje_vial = eje_vial
                    st.session_state.rasante = rasante
                    st.session_state.pendientes_tramos = pendientes_tramos
                    st.session_state.longitud_total = rasante["longitud_total"]
                    st.session_state.excavacion = None
                    st.session_state.pdf_memoria = None

                    if rasante.get("presupuesto_suficiente", False):
                        st.success("¡COMPLETADO! El presupuesto permite construir todo el eje.")
                    else:
                        st.warning(
                            "El presupuesto limita la construcción hasta "
                            f"{rasante['longitud_total']:.2f} m de "
                            f"{rasante['longitud_total_diseno']:.2f} m diseñados."
                        )

                if st.session_state.rasante is not None:
                    rasante = st.session_state.rasante

                    fig = crear_figura_maqueta(df, incluir_estratos=True)

                    agregar_franja_via(
                        fig=fig,
                        rasante=rasante,
                        ancho_via=ancho_via
                    )

                    fig.add_trace(go.Scatter3d(
                        x=rasante["xy"][:, 0],
                        y=rasante["xy"][:, 1],
                        z=rasante["z_terreno"] + 0.10,
                        mode="lines",
                        line=dict(color="yellow", width=3, dash="dot"),
                        name="Terreno natural bajo eje"
                    ))

                    agregar_tramos_coloreados_rasante(fig, rasante)

                    estaciones_estacas = np.arange(
                        0,
                        rasante["longitud_total"] + 100,
                        100
                    )

                    estaciones_estacas = estaciones_estacas[
                        estaciones_estacas <= rasante["longitud_total"]
                    ]

                    if len(estaciones_estacas) == 0 or estaciones_estacas[-1] < rasante["longitud_total"]:
                        estaciones_estacas = np.append(
                            estaciones_estacas,
                            rasante["longitud_total"]
                        )

                    xy_estacas = interpolar_eje(rasante["xy"], estaciones_estacas)

                    z_estacas = np.interp(
                        estaciones_estacas,
                        rasante["estaciones"],
                        rasante["z_rasante"]
                    )

                    fig.add_trace(go.Scatter3d(
                        x=xy_estacas[:, 0],
                        y=xy_estacas[:, 1],
                        z=z_estacas + 0.35,
                        mode="markers+text",
                        marker=dict(color="white", size=5),
                        text=[formato_km(e) for e in estaciones_estacas],
                        textposition="top center",
                        name="Estacas cada 100 m"
                    ))

                    agregar_lineas_verticales_rasante(
                        fig=fig,
                        rasante=rasante,
                        estaciones=estaciones_estacas
                    )

                    fig.add_trace(go.Scatter3d(
                        x=[rasante["xy"][0, 0]],
                        y=[rasante["xy"][0, 1]],
                        z=[rasante["z_rasante"][0] + 0.35],
                        mode="markers+text",
                        marker=dict(color="lime", size=12),
                        text=["INICIO"],
                        textposition="top center",
                        name="Inicio"
                    ))

                    fig.add_trace(go.Scatter3d(
                        x=[rasante["xy"][-1, 0]],
                        y=[rasante["xy"][-1, 1]],
                        z=[rasante["z_rasante"][-1] + 0.35],
                        mode="markers+text",
                        marker=dict(color="red", size=12, symbol="x"),
                        text=["FINAL"],
                        textposition="top center",
                        name="Final"
                    ))

                    fig.update_layout(
                        height=850,
                        template="plotly_dark",
                        scene=dict(
                            xaxis_title="X",
                            yaxis_title="Y",
                            zaxis_title="Z",
                            bgcolor="rgb(5,5,5)",
                            aspectmode="manual",
                            aspectratio=dict(x=1.6, y=1.0, z=0.60),
                            camera=dict(eye=dict(x=1.7, y=1.7, z=0.95))
                        ),
                        margin=dict(l=0, r=0, b=0, t=0),
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

        elif opcion == "Maqueta de Excavación 3D":
            st.subheader("⛏️ Maqueta de Excavación 3D")

            if st.session_state.rasante is None:
                st.warning("Primero calcula el eje y rasante en la Fase 7.")

            elif st.session_state.parametros_diseno is None:
                st.warning("Primero guarda los parámetros de diseño en la Fase 6.")

            else:
                rasante = st.session_state.rasante
                parametros = st.session_state.parametros_diseno
                ancho_via = parametros["ancho_via"]

                st.info("Maqueta de excavación basada en la rasante calculada en la Fase 7.")

                fig = crear_figura_maqueta(df, incluir_estratos=True)

                borde_izq, borde_der = calcular_bordes_via(rasante, ancho_via)

                volumen_corte, volumen_relleno = agregar_corte_relleno_3d(
                    fig=fig,
                    rasante=rasante,
                    ancho_via=ancho_via
                )

                agregar_zanja_excavacion(
                    fig=fig,
                    rasante=rasante,
                    ancho_via=ancho_via
                )

                st.session_state.excavacion = {
                    "longitud_construida": rasante["longitud_total"],
                    "longitud_total_diseno": rasante.get("longitud_total_diseno", rasante["longitud_total"]),
                    "volumen_corte": volumen_corte,
                    "volumen_relleno": volumen_relleno,
                    "volumen_total": volumen_corte + volumen_relleno,
                    "ancho_via": ancho_via,
                    "presupuesto": parametros["presupuesto_tierra"],
                    "presupuesto_suficiente": rasante.get("presupuesto_suficiente", True)
                }

                fig.add_trace(go.Scatter3d(
                    x=rasante["xy"][:, 0],
                    y=rasante["xy"][:, 1],
                    z=rasante["z_terreno"] + 0.10,
                    mode="lines",
                    line=dict(color="yellow", width=4, dash="dot"),
                    name="Eje Natural (Proyectado)"
                ))

                fig.add_trace(go.Scatter3d(
                    x=rasante["xy"][:, 0],
                    y=rasante["xy"][:, 1],
                    z=rasante["z_rasante"] + 0.18,
                    mode="lines",
                    line=dict(color="blue", width=9),
                    name="Eje Construido"
                ))

                fig.add_trace(go.Scatter3d(
                    x=borde_der[:, 0],
                    y=borde_der[:, 1],
                    z=rasante["z_rasante"] + 0.20,
                    mode="lines",
                    line=dict(color="orange", width=5, dash="dot"),
                    name="Derecho Vía"
                ))

                fig.add_trace(go.Scatter3d(
                    x=borde_izq[:, 0],
                    y=borde_izq[:, 1],
                    z=rasante["z_rasante"] + 0.20,
                    mode="lines",
                    line=dict(color="orange", width=5, dash="dot"),
                    name="Izquierdo Vía"
                ))

                fig.add_trace(go.Scatter3d(
                    x=[rasante["xy"][0, 0]],
                    y=[rasante["xy"][0, 1]],
                    z=[rasante["z_rasante"][0] + 0.45],
                    mode="markers+text",
                    marker=dict(color="fuchsia", size=10, symbol="diamond"),
                    text=["K0+000"],
                    textposition="top center",
                    name="Estaca 0+000"
                ))

                fig.add_trace(go.Scatter3d(
                    x=[rasante["xy"][-1, 0]],
                    y=[rasante["xy"][-1, 1]],
                    z=[rasante["z_rasante"][-1] + 0.45],
                    mode="markers+text",
                    marker=dict(color="red", size=13, symbol="x"),
                    text=["META"],
                    textposition="top center",
                    name="Llegada Meta"
                ))

                estaciones_estacas = np.arange(
                    0,
                    rasante["longitud_total"] + 100,
                    100
                )

                estaciones_estacas = estaciones_estacas[
                    estaciones_estacas <= rasante["longitud_total"]
                ]

                if len(estaciones_estacas) == 0 or estaciones_estacas[-1] < rasante["longitud_total"]:
                    estaciones_estacas = np.append(
                        estaciones_estacas,
                        rasante["longitud_total"]
                    )

                xy_estacas = interpolar_eje(rasante["xy"], estaciones_estacas)

                z_estacas = np.interp(
                    estaciones_estacas,
                    rasante["estaciones"],
                    rasante["z_rasante"]
                )

                fig.add_trace(go.Scatter3d(
                    x=xy_estacas[:, 0],
                    y=xy_estacas[:, 1],
                    z=z_estacas + 0.35,
                    mode="markers+text",
                    marker=dict(color="white", size=5),
                    text=[formato_km(e) for e in estaciones_estacas],
                    textposition="top center",
                    name="Estacas"
                ))

                fig.update_layout(
                    height=850,
                    template="plotly_dark",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        bgcolor="rgb(5,5,5)",
                        aspectmode="manual",
                        aspectratio=dict(x=1.6, y=1.0, z=0.60),
                        camera=dict(eye=dict(x=1.7, y=1.7, z=0.95))
                    ),
                    margin=dict(l=0, r=0, b=0, t=0),
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📋 Resumen Oficial de Obra")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Longitud construida real",
                    f"{rasante['longitud_total']:.2f} m"
                )

                col2.metric(
                    "Volumen corte acumulado",
                    f"{volumen_corte:.2f} m³"
                )

                col3.metric(
                    "Volumen relleno acumulado",
                    f"{volumen_relleno:.2f} m³"
                )

        elif opcion == "Base de Datos (Archivero)":
            st.subheader("Fase 9: Archivero de Diseños (SQLite3)")
            st.info("Guarda el historial de tus cálculos volumétricos para compararlos.")

            if st.session_state.excavacion is None:
                st.warning("Primero calcula la maqueta de excavación en la Fase 8.")

            else:
                excavacion = st.session_state.excavacion

                nombre_simulacion = st.text_input(
                    "Ingresa un nombre para guardar esta simulación"
                )

                if st.button("Guardar iteración actual en la Base de Datos"):
                    if nombre_simulacion.strip() == "":
                        st.error("Ingresa un nombre para la simulación.")
                    else:
                        guardar_simulacion(
                            nombre=nombre_simulacion.strip(),
                            ancho=excavacion["ancho_via"],
                            presupuesto=excavacion["presupuesto"],
                            longitud=excavacion["longitud_construida"],
                            corte=excavacion["volumen_corte"],
                            relleno=excavacion["volumen_relleno"]
                        )

                        st.success("Iteración guardada con éxito.")

                simulaciones = leer_simulaciones()

                if simulaciones.empty:
                    st.warning("Todavía no existen simulaciones guardadas.")
                else:
                    st.dataframe(simulaciones, use_container_width=True)

        elif opcion == "Emisión de Memoria (PDF)":
            st.subheader("Fase 10: Memoria de Cálculo Legal (fpdf2)")
            st.info("Adjunta la captura fotográfica del modelo 3D y emite el PDF formal.")

            if FPDF is None:
                st.error("Falta instalar fpdf2. Ejecuta: pip install fpdf2")

            elif st.session_state.excavacion is None:
                st.warning("Primero calcula la maqueta de excavación en la Fase 8.")

            else:
                simulaciones = leer_simulaciones()

                if simulaciones.empty:
                    st.warning("No hay simulaciones guardadas en la base de datos. Guarda al menos una en la Fase 9.")

                nombre_proyecto = st.text_input("Nombre del Proyecto")

                capturas = st.file_uploader(
                    "Sube la captura de pantalla de tu Maqueta (PNG/JPG)",
                    type=["png", "jpg", "jpeg"],
                    accept_multiple_files=True
                )

                if st.button("Generar Memoria de Cálculo PDF"):
                    if nombre_proyecto.strip() == "":
                        st.error("Ingresa el nombre del proyecto.")
                    elif not capturas:
                        st.error("Sube al menos una captura PNG/JPG.")
                    else:
                        excavacion = st.session_state.excavacion

                        resumen = {
                            "ancho": excavacion["ancho_via"],
                            "presupuesto": excavacion["presupuesto"],
                            "longitud": excavacion["longitud_construida"],
                            "corte": excavacion["volumen_corte"],
                            "relleno": excavacion["volumen_relleno"]
                        }

                        pdf_bytes = generar_memoria_pdf(
                            nombre_proyecto=nombre_proyecto.strip(),
                            resumen=resumen,
                            simulaciones=simulaciones,
                            capturas=capturas
                        )

                        st.session_state.pdf_memoria = pdf_bytes
                        st.success("Memoria de cálculo PDF generada correctamente.")

                if st.session_state.pdf_memoria is not None:
                    st.download_button(
                        label="Descargar Memoria de Cálculo PDF",
                        data=st.session_state.pdf_memoria,
                        file_name="memoria_calculo_diseno_vial.pdf",
                        mime="application/pdf"
                    )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Por favor, cargue un archivo.")