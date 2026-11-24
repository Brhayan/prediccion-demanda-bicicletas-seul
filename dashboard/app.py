# -*- coding: utf-8 -*-
"""Tablero interactivo de la demanda de bicicletas compartidas en Seúl.

Fase 4 del proyecto integrador. Reúne cuatro módulos: filtros dinámicos sobre el
histórico, tarjetas de indicadores, predicción en vivo con los modelos de la
Fase 3 y el mapa de los clústeres meteorológicos.

Arranque local:

    streamlit run dashboard/app.py

No entrena nada. Carga los modelos ya ajustados de `models/` y construye las
variables de entrada con `src/preparacion.py`, el mismo código con el que se
prepararon los datos de entrenamiento.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.preparacion import (CLASES, ESTACIONES, ESTACIONES_ES,  # noqa: E402
                             NUMERICAS, OBJETIVO, cortes_de_clase,
                             fila_de_entrada, nivel_demanda, particionar,
                             preparar)

MODELOS = RAIZ / 'models'

st.set_page_config(page_title='Demanda de bicicletas en Seúl',
                   page_icon='🚲', layout='wide')

# --------------------------------------------------------------------------
# Identidad visual. Es la misma de los cuadernos: el tablero y el informe deben
# verse como piezas de un mismo trabajo.
# --------------------------------------------------------------------------
AZUL, NARANJA, AQUA, AMBAR = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
TINTA, TINTA_TENUE, GRIS = '#0b0b0b', '#52514e', '#f0efec'
AZUL_OSCURO, AZUL_CLARO = '#0d366b', '#cde2fb'

ESCALA_SECUENCIAL = [[0.00, '#eef5fd'], [0.20, AZUL_CLARO], [0.40, '#86b6ef'],
                     [0.60, '#3987e5'], [0.80, '#1c5cab'], [1.00, AZUL_OSCURO]]

eje = dict(showline=True, linecolor=TINTA_TENUE, linewidth=1, ticks='outside',
           tickcolor=TINTA_TENUE, ticklen=4, zeroline=False,
           tickfont=dict(size=11, color=TINTA_TENUE),
           title=dict(font=dict(size=12, color=TINTA_TENUE)))

pio.templates['proyecto_bicicletas'] = go.layout.Template(layout=go.Layout(
    colorway=[AZUL, NARANJA, AQUA, AMBAR],
    font=dict(family='Segoe UI, Helvetica, Arial, sans-serif', size=12, color=TINTA),
    title=dict(font=dict(size=16, color=TINTA), x=0, xanchor='left'),
    paper_bgcolor='white', plot_bgcolor='white',
    xaxis={**eje, 'showgrid': False},
    yaxis={**eje, 'showgrid': True, 'gridcolor': GRIS, 'gridwidth': 1},
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left',
                x=0, title_text=''),
    margin=dict(l=70, r=30, t=70, b=60),
    hoverlabel=dict(bgcolor='white', bordercolor=TINTA_TENUE,
                    font=dict(size=12, color=TINTA)),
    colorscale=dict(sequential=ESCALA_SECUENCIAL)))
pio.templates.default = 'proyecto_bicicletas'

# Los titulos de las figuras no van dentro de Plotly sino en un `st.markdown`
# con `help`: es la unica forma de colgarles el mismo signo de interrogacion que
# llevan las tarjetas, porque `st.plotly_chart` no acepta ayuda emergente.
def titulo(texto: str, ayuda: str) -> None:
    st.markdown(f'**{texto}**', help=ayuda)


COLOR_GRUPO = {0: AZUL, 1: AMBAR, 2: AQUA}
COLOR_CLASE = {'Baja': AZUL_CLARO, 'Media': AZUL, 'Alta': AZUL_OSCURO}


# --------------------------------------------------------------------------
# Carga. Los modelos ocupan memoria y se comparten entre sesiones
# (cache_resource); las tablas se copian por sesión (cache_data).
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner='Cargando los modelos de la Fase 3...')
def cargar_modelos() -> dict:
    with open(MODELOS / 'metadatos.json', encoding='utf-8') as f:
        metadatos = json.load(f)
    return {
        'regresion': joblib.load(MODELOS / 'regresion_xgboost.joblib'),
        'clasificacion': joblib.load(MODELOS / 'clasificacion_xgboost.joblib'),
        'segmentacion': joblib.load(MODELOS / 'segmentacion_kmeans.joblib'),
        'metadatos': metadatos,
    }


@st.cache_data(show_spinner='Preparando el histórico...')
def cargar_historico() -> tuple[pd.DataFrame, dict]:
    """Reconstruye las 8465 horas y les añade nivel de demanda y clúster.

    Parte del archivo original, no de los CSV intermedios, porque esos no se
    versionan: el tablero desplegado solo encuentra `data/raw/`.
    """
    datos = preparar()
    bloques = particionar(datos)
    corte_bajo, corte_alto = cortes_de_clase(bloques['entrenamiento'][OBJETIVO])

    datos['nivel_demanda'] = nivel_demanda(datos[OBJETIVO], corte_bajo, corte_alto)
    datos['estacion'] = datos['Seasons'].map(ESTACIONES_ES)

    modelos = cargar_modelos()
    seg = modelos['segmentacion']
    Z = seg['escalador'].transform(datos[NUMERICAS])
    datos['grupo'] = seg['kmeans'].predict(Z)
    coords = seg['pca'].transform(Z)
    datos['pca_1'], datos['pca_2'] = coords[:, 0], coords[:, 1]

    etiquetas = {int(g): e for g, e in modelos['metadatos']['etiquetas_grupo'].items()}
    datos['grupo_nombre'] = datos['grupo'].map(etiquetas)

    contexto = {
        'corte_bajo': corte_bajo, 'corte_alto': corte_alto,
        'etiquetas': etiquetas,
        'varianza_pca': seg['pca'].explained_variance_ratio_,
        # Se reconstruye desde `datos` para que arrastre el nivel de demanda y
        # el grupo, que se calcularon despues de particionar.
        'prueba': datos.loc[bloques['prueba'].index],
        'horas_entrenamiento': len(bloques['entrenamiento']),
    }
    return datos, contexto


@st.cache_data(show_spinner=False)
def medir_en_prueba(_prueba: pd.DataFrame) -> dict:
    """Recalcula las métricas sobre el conjunto de prueba al arrancar.

    No se escriben a mano: si el tablero cargara unos modelos distintos de los
    que evaluó el cuaderno 03, las tarjetas lo delatarían en el acto.
    """
    from sklearn.metrics import f1_score, mean_absolute_error, r2_score

    modelos = cargar_modelos()
    y = _prueba[OBJETIVO].to_numpy(dtype=float)
    pred = modelos['regresion'].predict(_prueba)
    clase_real = _prueba['nivel_demanda'].astype(str).map({c: i for i, c in enumerate(CLASES)})
    clase_pred = modelos['clasificacion'].predict(_prueba)
    return {
        'r2': r2_score(y, pred),
        'mae': mean_absolute_error(y, pred),
        'f1': f1_score(clase_real, clase_pred, average='macro'),
    }


datos, contexto = cargar_historico()
modelos = cargar_modelos()
metricas = medir_en_prueba(contexto['prueba'])


# --------------------------------------------------------------------------
# Módulo 1. Filtros dinámicos
# --------------------------------------------------------------------------
st.sidebar.title('Filtros')
st.sidebar.caption('Afectan a los indicadores, al panorama y al mapa de clústeres. '
                   'La predicción en vivo usa sus propios controles.')

fecha_min = datos['fecha'].min().date()
fecha_max = datos['fecha'].max().date()
rango = st.sidebar.date_input('Rango de fechas', value=(fecha_min, fecha_max),
                              min_value=fecha_min, max_value=fecha_max)
if isinstance(rango, tuple) and len(rango) == 2:
    desde, hasta = rango
else:
    desde = hasta = rango if not isinstance(rango, tuple) else rango[0]

estaciones = st.sidebar.multiselect(
    'Estaciones del año', options=[ESTACIONES_ES[e] for e in ESTACIONES],
    default=[ESTACIONES_ES[e] for e in ESTACIONES])

hora_desde, hora_hasta = st.sidebar.slider('Franja horaria', 0, 23, (0, 23))

t_min, t_max = float(datos['Temperature(°C)'].min()), float(datos['Temperature(°C)'].max())
temp_desde, temp_hasta = st.sidebar.slider(
    'Temperatura (°C)', t_min, t_max, (t_min, t_max), step=0.5)

solo_laborables = st.sidebar.checkbox('Solo días laborables', value=False)

sel = datos[
    (datos['fecha'].dt.date >= desde) & (datos['fecha'].dt.date <= hasta)
    & (datos['estacion'].isin(estaciones))
    & (datos['Hour'].between(hora_desde, hora_hasta))
    & (datos['Temperature(°C)'].between(temp_desde, temp_hasta))
]
if solo_laborables:
    sel = sel[(sel['es_fin_de_semana'] == 'No') & (sel['Holiday'] == 'No Holiday')]

st.sidebar.markdown(f'**{len(sel)} horas** de {len(datos)} seleccionadas')


# --------------------------------------------------------------------------
# Encabezado y módulo 2. Tarjetas de indicadores
# --------------------------------------------------------------------------
st.title('Demanda horaria de bicicletas compartidas en Seúl')
st.caption('Proyecto integrador de Ciencia de Datos · Maestría en Inteligencia Artificial, '
           'Universidad de La Salle · Brhayan Liberato Tafur')

if sel.empty:
    st.warning('Ninguna hora cumple los filtros seleccionados. Amplíe el rango en la barra lateral.')
    st.stop()

perfil_horario = sel.groupby('Hour')[OBJETIVO].mean()
hora_pico = int(perfil_horario.idxmax())

st.subheader('Demanda del servicio de alquiler', divider='blue')

c1, c2, c3, c4 = st.columns(4)
c1.metric('Demanda media por hora', f'{sel[OBJETIVO].mean():.0f}',
          border=True,
          help='Promedio de bicicletas alquiladas en las horas seleccionadas.')
c2.metric('Hora de mayor demanda', f'{hora_pico:02d}:00',
          f'{perfil_horario.max():.0f} en promedio', border=True,
          help='Hora del día con el promedio más alto entre las horas filtradas.')
c3.metric('Máximo registrado', f'{sel[OBJETIVO].max():.0f}', border=True)
c4.metric('Horas de alta demanda',
          f'{(sel["nivel_demanda"] == "Alta").mean():.0%}', border=True,
          help=f'Por encima de {contexto["corte_alto"]:.0f} bicicletas por hora.')

st.divider()

# Las etiquetas van en negrita y cada pestana abre con su propio encabezado: son
# los tres modulos del tablero y deben leerse como pantallas distintas.
# Streamlit no expone el tamano de letra de las pestanas, asi que se agranda con
# CSS para que el menu se distinga del texto del cuerpo.
st.markdown(
    '''
    <style>
    .stTabs [role="tablist"] { gap: 1.75rem; }
    .stTabs [data-testid="stTab"] { padding: 0.6rem 0.25rem; }
    .stTabs [data-testid="stTab"] p,
    .stTabs [data-testid="stTab"] strong { font-size: 1.3rem !important; }
    </style>
    ''',
    unsafe_allow_html=True,
)
panorama, prediccion, segmentacion = st.tabs(
    ['**Panorama**', '**Predicción en vivo**', '**Segmentación**'])


# --------------------------------------------------------------------------
# Panorama
# --------------------------------------------------------------------------
with panorama:
    st.header('Panorama', divider='blue')
    st.caption('Cómo se comporta la demanda en las horas que dejan pasar los filtros.')

    izq, der = st.columns(2)

    with izq:
        por_estacion = (sel.groupby(['Hour', 'estacion'])[OBJETIVO].mean()
                        .reset_index())
        fig = go.Figure()
        for i, est in enumerate([ESTACIONES_ES[e] for e in ESTACIONES]):
            bloque = por_estacion[por_estacion['estacion'] == est]
            if bloque.empty:
                continue
            fig.add_trace(go.Scatter(
                x=bloque['Hour'], y=bloque[OBJETIVO], mode='lines',
                name=est, line=dict(width=2.5),
                hovertemplate='%{y:.0f} bicicletas a las %{x}:00<extra>' + est + '</extra>'))
        titulo('Perfil horario medio por estación del año',
               'Cada línea es una estación del año y recorre las 24 horas del día. '
               'La altura marca cuántas bicicletas se alquilaron en promedio a esa '
               'hora. Los dos picos son las entradas y salidas del trabajo. Cuanto '
               'más separadas van las líneas, más pesa la estación frente a la hora.')
        fig.update_layout(
            xaxis_title='Hora del día', yaxis_title='Bicicletas por hora',
            xaxis=dict(tickmode='array', tickvals=list(range(0, 24, 3))),
            height=400, margin=dict(l=70, r=30, t=40, b=60))
        st.plotly_chart(fig, width='stretch')

    with der:
        fig = go.Figure(go.Scattergl(
            x=sel['Temperature(°C)'], y=sel[OBJETIVO], mode='markers',
            marker=dict(size=4, color=sel['Humidity(%)'],
                        colorscale=ESCALA_SECUENCIAL, opacity=0.55,
                        colorbar=dict(title='Humedad<br>(%)', thickness=12)),
            hovertemplate='%{y:.0f} bicicletas<br>%{x:.1f} °C<extra></extra>'))
        titulo('Demanda frente a temperatura, coloreada por humedad',
               'Cada punto es una hora del histórico. Hacia la derecha hace más '
               'calor y hacia arriba se alquilaron más bicicletas. El color indica '
               'la humedad: los puntos oscuros, los más húmedos, se acumulan abajo, '
               'y esa es la señal de que la humedad frena la demanda incluso cuando '
               'la temperatura acompaña.')
        fig.update_layout(
            xaxis_title='Temperatura (°C)', yaxis_title='Bicicletas por hora',
            height=400, margin=dict(l=70, r=30, t=15, b=60))
        st.plotly_chart(fig, width='stretch')

    reparto = (sel.groupby('estacion', observed=True)
               .agg(horas=(OBJETIVO, 'size'), demanda_media=(OBJETIVO, 'mean'),
                    maximo=(OBJETIVO, 'max'), temperatura=('Temperature(°C)', 'mean'))
               .round(1).sort_values('demanda_media', ascending=False))
    titulo('Resumen por estación del año',
           'Una fila por estación, ordenadas de mayor a menor demanda media. '
           '`horas` cuenta los registros que dejan pasar los filtros, de modo que '
           'una estación con pocas horas tiene promedios menos firmes.')
    st.dataframe(reparto, width='stretch')


# --------------------------------------------------------------------------
# Módulo 3. Predicción en vivo
# --------------------------------------------------------------------------
with prediccion:
    st.header('Predicción en vivo', divider='blue')
    st.markdown('Introduzca unas condiciones y los modelos de la Fase 3 devuelven '
                'la demanda esperada para esa hora. El punto de rocío que necesitan '
                'se deriva de la temperatura y la humedad.')

    # Las tres cifras acompanan al formulario, no al panel de la izquierda: ahi
    # responden a la pregunta que surge al ver una prediccion, y ademas son fijas,
    # mientras que las tarjetas de demanda cambian con cada filtro.
    st.markdown('**Qué tan fiable es lo que verá abajo**')
    m1, m2, m3 = st.columns(3)
    m1.metric('R² del modelo de regresión', f'{metricas["r2"]:.4f}', border=True,
              help='XGBoost sobre el conjunto de prueba, nunca usado para ajustar.')
    m2.metric('Error medio absoluto', f'{metricas["mae"]:.0f} bicicletas', border=True,
              help='Cuánto se desvía en promedio la demanda estimada.')
    m3.metric('F1 macro de la clasificación', f'{metricas["f1"]:.4f}', border=True,
              help='Promedio del F1 de las tres clases, sin favorecer a la mayoritaria.')
    st.caption(f'Los modelos se entrenaron con {contexto["horas_entrenamiento"]} horas '
               'y estas tres cifras se '
               'recalculan en cada arranque sobre el conjunto de prueba, que ningún '
               'modelo vio durante el ajuste. No cambian con los filtros de la izquierda.')

    st.divider()

    with st.form('formulario'):
        f1, f2, f3 = st.columns(3)
        with f1:
            temperatura = st.slider('Temperatura (°C)', -20.0, 40.0, 21.0, 0.5)
            humedad = st.slider('Humedad relativa (%)', 10, 100, 45)
            viento = st.slider('Viento (m/s)', 0.0, 8.0, 1.8, 0.1)
        with f2:
            visibilidad = st.slider('Visibilidad (decámetros)', 27, 2000, 1800, 10)
            radiacion = st.slider('Radiación solar (MJ/m²)', 0.0, 3.6, 0.4, 0.05)
            hora = st.slider('Hora del día', 0, 23, 18)
        with f3:
            estacion = st.selectbox(
                'Estación del año', options=ESTACIONES,
                format_func=lambda e: ESTACIONES_ES[e], index=2)
            es_fin_de_semana = st.checkbox('Fin de semana')
            es_festivo = st.checkbox('Día festivo')
            llueve = st.checkbox('Está lloviendo')
            nieva = st.checkbox('Está nevando')
        enviar = st.form_submit_button('Calcular la predicción', type='primary')

    if enviar:
        fila = fila_de_entrada(
            temperatura=temperatura, humedad=humedad, viento=viento,
            visibilidad=visibilidad, radiacion=radiacion, hora=hora,
            estacion=estacion, es_festivo=es_festivo,
            es_fin_de_semana=es_fin_de_semana, llueve=llueve, nieva=nieva)

        conteo = float(modelos['regresion'].predict(fila)[0])
        probabilidades = modelos['clasificacion'].predict_proba(fila)[0]
        clase = CLASES[int(np.argmax(probabilidades))]

        seg = modelos['segmentacion']
        Z = seg['escalador'].transform(fila[NUMERICAS])
        grupo = int(seg['kmeans'].predict(Z)[0])

        r1, r2 = st.columns(2)
        r1.metric('Demanda estimada', f'{max(conteo, 0):.0f} bicicletas',
                  border=True, help='XGBoost de regresión.')
        r2.metric('Nivel de demanda', clase,
                  f'{probabilidades[CLASES.index(clase)]:.1%} de confianza', border=True)

        fig = go.Figure(go.Bar(
            x=[float(p) for p in probabilidades], y=CLASES, orientation='h',
            marker_color=[COLOR_CLASE[c] for c in CLASES],
            text=[f'{p:.1%}' for p in probabilidades], textposition='outside',
            hovertemplate='%{y}: %{x:.1%}<extra></extra>'))
        titulo('Probabilidad asignada a cada nivel',
               'El modelo reparte el 100 % entre los tres niveles de demanda. La '
               'barra más larga es el nivel que escoge. Si las tres barras se '
               'parecen, la predicción es dudosa y conviene decirlo; si una se '
               'lleva casi todo, el caso es claro para el modelo.')
        fig.update_layout(
            xaxis=dict(range=[0, 1.12], tickformat='.0%', title='Probabilidad'),
            yaxis=dict(categoryorder='array', categoryarray=CLASES[::-1]),
            height=230, margin=dict(l=70, r=30, t=15, b=40), showlegend=False)
        st.plotly_chart(fig, width='stretch')

        st.info(f'Estas condiciones pertenecen al grupo meteorológico '
                f'**{contexto["etiquetas"][grupo]}**, cuyo perfil se describe en la '
                f'pestaña de segmentación.')

        with st.expander('Ver la fila que reciben los modelos'):
            # Todo a texto: la tabla mezcla numeros y categorias, y Arrow no
            # sabe serializar una columna de tipos mixtos.
            resumen = pd.DataFrame({
                'variable': list(fila.columns),
                'valor': [f'{v:.4f}' if isinstance(v, float) else str(v)
                          for v in fila.iloc[0]]})
            st.dataframe(resumen, width='stretch', hide_index=True)
            st.caption('Las trece columnas, en el mismo orden con el que se ajustó el '
                       'preprocesador. Las derivadas las construye `src/preparacion.py`.')


# --------------------------------------------------------------------------
# Módulo 4. Segmentación
# --------------------------------------------------------------------------
with segmentacion:
    st.header('Segmentación', divider='blue')
    st.markdown('K-Means con K = 3 sobre las ocho variables numéricas estandarizadas, '
                'ajustado con el conjunto de entrenamiento. Cada punto es una hora, '
                'situada en el plano de las dos primeras componentes principales.')

    izq, der = st.columns([3, 2])

    with izq:
        v1, v2 = contexto['varianza_pca']
        fig = go.Figure()
        for g, nombre in contexto['etiquetas'].items():
            bloque = sel[sel['grupo'] == g]
            if bloque.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=bloque['pca_1'], y=bloque['pca_2'], mode='markers',
                name=f'{nombre} ({len(bloque)} h)',
                marker=dict(size=4, color=COLOR_GRUPO[g], opacity=0.45, line_width=0),
                customdata=bloque[OBJETIVO],
                hovertemplate='<b>%{customdata:.0f}</b> bicicletas<extra>'
                              + nombre + '</extra>'))
        titulo('Clústeres meteorológicos en el espacio de componentes principales',
               'Las ocho variables meteorológicas se comprimen en dos ejes para '
               'poder dibujarlas en un plano. Esos ejes no tienen unidades y no se '
               'leen como grados ni como bicicletas: lo que importa es la cercanía, '
               'porque dos puntos próximos son horas de clima parecido. El color '
               'indica a qué grupo asignó K-Means cada hora.')
        fig.update_layout(
            xaxis_title=f'Componente 1 ({v1:.1%} de la varianza)',
            yaxis_title=f'Componente 2 ({v2:.1%} de la varianza)',
            height=490, margin=dict(l=70, r=30, t=40, b=60))
        st.plotly_chart(fig, width='stretch')

    with der:
        perfiles = (sel.groupby('grupo_nombre', observed=True)
                    .agg(horas=(OBJETIVO, 'size'),
                         temperatura=('Temperature(°C)', 'mean'),
                         humedad=('Humidity(%)', 'mean'),
                         radiacion=('Solar Radiation (MJ/m2)', 'mean'),
                         demanda=(OBJETIVO, 'mean'))
                    .round(1).sort_values('demanda', ascending=False))
        titulo('Perfil numérico medio de cada grupo',
               'El promedio de cada variable dentro del grupo. Es lo que da sentido '
               'a los nombres: el grupo húmedo y nublado ronda el 79 % de humedad '
               'con radiación casi nula. Los nombres resumen tres variables, pero el '
               'agrupamiento usó ocho, así que alguna hora puede caer en un grupo '
               'que su nombre no haría esperar.')
        st.dataframe(perfiles, width='stretch')

        # El color se toma del nombre, no del orden: la tabla va ordenada por
        # demanda y los grupos por su numero, que no coinciden.
        grupo_de = {n: g for g, n in contexto['etiquetas'].items()}
        fig = go.Figure(go.Bar(
            x=perfiles['demanda'], y=perfiles.index, orientation='h',
            marker_color=[COLOR_GRUPO[grupo_de[n]] for n in perfiles.index],
            text=[f'{d:.0f}' for d in perfiles['demanda']],
            textposition='outside', hovertemplate='%{x:.0f} bicicletas<extra></extra>'))
        titulo('Demanda media por grupo',
               'Bicicletas por hora que registra cada grupo en promedio, sobre las '
               'horas filtradas. Los colores son los mismos del gráfico de la '
               'izquierda, de modo que una barra y su nube de puntos se corresponden.')
        fig.update_layout(
            xaxis_title='Bicicletas por hora',
            xaxis=dict(range=[0, perfiles['demanda'].max() * 1.25]),
            height=250, margin=dict(l=130, r=30, t=15, b=50), showlegend=False)
        st.plotly_chart(fig, width='stretch')

st.divider()
st.caption('Los modelos no se entrenan aquí: se cargan ya ajustados desde `models/`, '
           'tal como los dejó el cuaderno `03_modelado_avanzado.ipynb`. '
           'Datos: Seoul Bike Sharing Demand, UCI Machine Learning Repository, licencia CC BY 4.0.')
