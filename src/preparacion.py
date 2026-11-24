# -*- coding: utf-8 -*-
"""Preparación de los datos, compartida entre los cuadernos y el tablero.

Los cuadernos son autocontenidos y no importan este módulo: cada entregable se
corrige por separado. El tablero de la Fase 4 sí lo hace, y por una razón
concreta. El formulario de predicción en vivo tiene que construir exactamente
las mismas variables con las que se entrenaron los modelos de `models/`, porque
una diferencia en la fórmula de `hora_sin` o en el criterio de `llueve` produce
una predicción silenciosamente equivocada. Duplicar esas reglas en el código del
tablero sería invitar a que se desincronicen.

Lo que hay aquí es, por tanto, una copia de lo que hace el cuaderno 02,
comprobada fila a fila contra las particiones que ese cuaderno escribió en
`data/processed/`: mismas 5925, 1270 y 1270 filas, mismo orden, mismas ocho
variables numéricas y mismos cortes de clase.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CRUDO = RAIZ / 'data' / 'raw' / 'SeoulBikeData.csv'

OBJETIVO = 'Rented Bike Count'
CLASES = ['Baja', 'Media', 'Alta']
SEMILLA = 42

NUMERICAS = ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
             'Visibility (10m)', 'Solar Radiation (MJ/m2)',
             'depresion_punto_rocio', 'hora_sin', 'hora_cos']
CATEGORICAS = ['Seasons', 'Holiday', 'es_fin_de_semana', 'llueve', 'nieva']
ENTRADAS = NUMERICAS + CATEGORICAS

ESTACIONES = ['Winter', 'Spring', 'Summer', 'Autumn']
ESTACIONES_ES = {'Winter': 'Invierno', 'Spring': 'Primavera',
                 'Summer': 'Verano', 'Autumn': 'Otoño'}


# --------------------------------------------------------------------------
# Carga y limpieza
# --------------------------------------------------------------------------
def cargar_crudo(ruta: Path | str = RUTA_CRUDO) -> pd.DataFrame:
    """Lee el archivo original. Exige cp1252 por el símbolo de grado."""
    return pd.read_csv(ruta, encoding='cp1252')


def excluir_horas_sin_servicio(crudo: pd.DataFrame) -> pd.DataFrame:
    """Descarta las 295 horas con el sistema cerrado.

    La demanda registrada en ellas es cero por ausencia de oferta, no de
    demanda: conservarlas enseñaría al modelo a predecir cero bajo condiciones
    meteorológicas que no lo causan. Tras excluirlas, `Functioning Day` toma un
    único valor y deja de aportar información.
    """
    abierto = crudo['Functioning Day'] != 'No'
    return (crudo.loc[abierto]
            .drop(columns=['Functioning Day'])
            .reset_index(drop=True))


def crear_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las seis variables derivadas de la Fase 2."""
    out = df.copy()
    fecha = pd.to_datetime(out['Date'], format='%d/%m/%Y')

    out['hora_sin'] = np.sin(2 * np.pi * out['Hour'] / 24)
    out['hora_cos'] = np.cos(2 * np.pi * out['Hour'] / 24)

    es_sabado_o_domingo = fecha.dt.dayofweek >= 5
    out['es_fin_de_semana'] = es_sabado_o_domingo.map({True: 'Si', False: 'No'})

    out['depresion_punto_rocio'] = (out['Temperature(°C)']
                                    - out['Dew point temperature(°C)'])

    out['llueve'] = np.where(out['Rainfall(mm)'] > 0, 'Si', 'No')
    out['nieva'] = np.where(out['Snowfall (cm)'] > 0, 'Si', 'No')

    return out


def preparar(ruta: Path | str = RUTA_CRUDO) -> pd.DataFrame:
    """Devuelve el conjunto de trabajo: 8465 horas con las variables derivadas."""
    datos = crear_variables_derivadas(excluir_horas_sin_servicio(cargar_crudo(ruta)))
    datos['fecha'] = pd.to_datetime(datos['Date'], format='%d/%m/%Y')
    return datos


# --------------------------------------------------------------------------
# Partición y objetivo de clasificación
# --------------------------------------------------------------------------
def particionar(datos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reproduce la partición 70/15/15 estratificada de la Fase 2.

    Devuelve un diccionario con los tres bloques completos, objetivo incluido.
    El estratificador es un corte en terciles sobre la demanda, el mismo que
    usó el cuaderno 02, de modo que con la misma semilla las filas coinciden.
    """
    estratos = pd.qcut(datos[OBJETIVO], q=3, labels=CLASES)

    ent, resto = train_test_split(
        datos, test_size=0.30, random_state=SEMILLA, stratify=estratos)
    val, pru = train_test_split(
        resto, test_size=0.50, random_state=SEMILLA,
        stratify=estratos.loc[resto.index])

    return {'entrenamiento': ent, 'validacion': val, 'prueba': pru}


def cortes_de_clase(y_entrenamiento: pd.Series) -> tuple[float, float]:
    """Terciles estimados **solo** con entrenamiento, para no filtrar datos."""
    bajo, alto = y_entrenamiento.quantile([1 / 3, 2 / 3])
    return float(bajo), float(alto)


def nivel_demanda(serie: pd.Series, corte_bajo: float, corte_alto: float) -> pd.Series:
    """Traduce una demanda horaria a Baja, Media o Alta."""
    return pd.cut(serie, bins=[-np.inf, corte_bajo, corte_alto, np.inf],
                  labels=CLASES)


# --------------------------------------------------------------------------
# Entrada del formulario de predicción en vivo
# --------------------------------------------------------------------------
def punto_de_rocio(temperatura: float, humedad: float) -> float:
    """Estima el punto de rocío en grados a partir de temperatura y humedad.

    Aproximación de Magnus-Tetens con los coeficientes de Alduchov y Eskridge
    (1996). El formulario no puede pedirle el punto de rocío a quien consulta el
    tablero, pero el modelo necesita la depresión. Contrastada contra las 8448
    horas del conjunto con humedad registrada: error absoluto medio de 0.06 °C y
    percentil 95 de 0.10 °C.
    """
    a, b = 17.62, 243.12
    humedad = min(max(float(humedad), 1.0), 100.0)
    gamma = np.log(humedad / 100.0) + a * temperatura / (b + temperatura)
    return float(b * gamma / (a - gamma))


def fila_de_entrada(temperatura: float, humedad: float, viento: float,
                    visibilidad: float, radiacion: float, hora: int,
                    estacion: str, es_festivo: bool, es_fin_de_semana: bool,
                    llueve: bool, nieva: bool) -> pd.DataFrame:
    """Compone la fila de una sola observación que los modelos esperan.

    Las claves son los nombres originales del archivo, con su espaciado
    irregular incluido, porque el `ColumnTransformer` serializado indexa por
    ellos. El orden de `ENTRADAS` es el mismo con el que se ajustó.
    """
    si_no = {True: 'Si', False: 'No'}
    fila = {
        'Temperature(°C)': float(temperatura),
        'Humidity(%)': float(humedad),
        'Wind speed (m/s)': float(viento),
        'Visibility (10m)': float(visibilidad),
        'Solar Radiation (MJ/m2)': float(radiacion),
        'depresion_punto_rocio': float(temperatura) - punto_de_rocio(temperatura, humedad),
        'hora_sin': float(np.sin(2 * np.pi * hora / 24)),
        'hora_cos': float(np.cos(2 * np.pi * hora / 24)),
        'Seasons': estacion,
        'Holiday': 'Holiday' if es_festivo else 'No Holiday',
        'es_fin_de_semana': si_no[bool(es_fin_de_semana)],
        'llueve': si_no[bool(llueve)],
        'nieva': si_no[bool(nieva)],
    }
    return pd.DataFrame([fila])[ENTRADAS]
