# Proyecto Integrador - Ciencia de Datos

Maestría en Inteligencia Artificial, Universidad de La Salle. Asignatura: Ciencia de Datos G01.

**Autor:** Brhayan Liberato Tafur

## Qué resuelve

Un sistema de bicicletas compartidas necesita saber con antelación cuántas bicicletas harán falta
en cada hora: redistribuir la flota cuesta dinero y solo se justifica si la demanda se anticipa.
Este proyecto predice la demanda horaria del sistema público de Seúl a partir de las condiciones
meteorológicas y del calendario, y lo hace sobre tres frentes a la vez:

- **Regresión:** cuántas bicicletas se alquilan en una hora dada.
- **Clasificación:** si esa hora será de demanda baja, media o alta.
- **Segmentación:** qué perfiles de operación existen a lo largo del año.

Sobre el conjunto de prueba, reservado desde el principio y usado una sola vez, el modelo final
alcanza un **R² de 0.9213** en regresión y un **F1 macro de 0.8967** en clasificación. Los modelos
alimentan un tablero interactivo que predice en vivo sobre condiciones que introduce el usuario.

## Conjunto de datos

Seoul Bike Sharing Demand - UCI Machine Learning Repository.
URL: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
Licencia: CC BY 4.0. 8760 registros horarios (01/12/2017 – 30/11/2018), 14 variables.

El archivo se lee con `encoding="cp1252"`, porque los nombres de columna llevan el símbolo de grado.

## Qué se hizo, fase por fase

### Fase 1 - Planteamiento y análisis exploratorio

`notebooks/01_planteamiento_y_eda.ipynb`

Formulación del problema, pregunta de investigación y objetivos, ficha técnica del conjunto con su
diccionario de datos, y un análisis exploratorio completo: estadísticos univariados con asimetría y
curtosis, correlaciones, pruebas χ² de independencia, diagnóstico de faltantes, atípicos y
duplicados.

Tres hallazgos condicionaron todo lo demás:

1. **Las 295 horas con demanda cero son exactamente las 295 horas con el sistema fuera de
   servicio.** Ese cero no mide ausencia de demanda sino de oferta.
2. **La temperatura y el punto de rocío están correlacionados a 0.91**, es decir, miden casi lo
   mismo.
3. **La relación de la hora con la demanda no es monótona**, tiene dos picos, y las correlaciones
   lineales la subestiman.

### Fase 2 - Preprocesamiento, variables derivadas y modelo base

`notebooks/02_preprocesamiento_y_baseline.ipynb`

Se excluyen las 295 horas sin servicio y el conjunto de trabajo queda en **8465 horas**. Se
construyen **seis variables derivadas**, entre ellas el seno y el coseno de la hora, que sitúan
contiguas la hora 23 y la hora 0, y la depresión del punto de rocío, que resuelve la redundancia
detectada en la Fase 1.

Los datos se parten en **70/15/15** estratificado por terciles de la demanda: 5925, 1270 y 1270
horas. Los cortes de clase y el escalado se calculan **solo con entrenamiento**, nunca con el
total, para que no se filtre información hacia validación ni prueba.

Se fijan los modelos base contra los que se mide todo lo posterior:

| Modelo base | Resultado |
| :--- | :--- |
| Trivial de regresión (predice la media) | R² -0.0007, RMSE 620.98 |
| Regresión lineal | R² 0.5454, RMSE 418.56 |
| Trivial de clasificación (clase mayoritaria) | F1 macro 0.1669 |
| Regresión logística | F1 macro 0.7225 |

### Fase 3 - Modelado avanzado, segmentación e interpretabilidad

`notebooks/03_modelado_avanzado.ipynb`

Búsqueda de hiperparámetros con validación cruzada de cinco pliegues para cada familia de modelos:

- **Regresión:** ElasticNet y XGBoost. El resultado más informativo es que la regularización no
  aporta nada, porque el problema es de sesgo y no de varianza; XGBoost sube el R² de 0.5454 a
  0.9297 en validación.
- **Clasificación:** Random Forest y XGBoost. Random Forest memoriza el entrenamiento completo
  (F1 = 1.0000), así que se conserva XGBoost, con una brecha cuatro veces menor.
- **Red neuronal:** un perceptrón multicapa en PyTorch (64-32-1, 150 épocas) que empata con XGBoost,
  R² 0.9282, lo que sugiere que el límite lo imponen los datos y no el algoritmo.
- **Segmentación:** K-Means con el codo y la silueta, más un agrupamiento jerárquico de Ward como
  comprobación. Tres grupos de operación con demanda media de 494, 744 y 1018 bicicletas por hora.
- **Interpretabilidad:** valores SHAP. La temperatura y el seno de la hora dominan, con 258.53 y
  208.20 de contribución media.

Al final de la fase se evalúa **una sola vez** sobre el conjunto de prueba y se serializan los
modelos elegidos en `models/`.

| Problema | Resultado sobre prueba |
| :--- | :--- |
| Regresión | R² 0.9213 · RMSE 181.06 · MAE 110.59 |
| Clasificación | Exactitud 0.8969 · F1 macro 0.8967 · AUC 0.9742 |
| Segmentación | 3 grupos · silueta 0.2201 · 59.4 % de varianza en dos componentes |

### Fase 4 - Tablero interactivo e informe

`dashboard/app.py` y `reports/informe_tecnico_fase4.pdf`

El tablero no entrena nada: carga los modelos ya ajustados y responde una predicción nueva en unos
100 milisegundos. Reúne filtros dinámicos, tarjetas de indicadores, un formulario de predicción en
vivo que devuelve la demanda estimada y el nivel esperado, y el mapa de los clústeres en el espacio
de componentes principales. El informe técnico recoge el trabajo completo con estructura de
artículo.

## Estructura

| Carpeta | Contenido |
| :--- | :--- |
| `data/raw/` | Conjunto de datos original, sin modificar |
| `data/processed/` | Particiones de la Fase 2. Se regeneran con el cuaderno 02 |
| `notebooks/` | Un cuaderno por fase, con su exportación a HTML |
| `src/` | Las reglas de preparación que el tablero comparte con los cuadernos |
| `models/` | Modelos serializados que consume el tablero. Los escribe el cuaderno 03 |
| `dashboard/` | Aplicación interactiva de la Fase 4 |
| `reports/` | Informe técnico en PDF |

## Reproducción

```bash
conda create -n ciencia_datos python=3.11.9
conda activate ciencia_datos
pip install -r requirements.txt
jupyter lab
```

Ejecutar los cuadernos de `notebooks/` en orden numérico. **Cada uno se ejecuta por sí solo**, sin
importar código de ningún otro archivo del repositorio; la única dependencia entre ellos son los
datos, porque el cuaderno 03 lee las particiones que el 02 escribe en `data/processed/`. Ejecutar
el 03 sin haber ejecutado el 02 termina en `FileNotFoundError`.

`data/raw/SeoulBikeData.csv` **debe estar presente antes de ejecutar nada**: es la única entrada
del proyecto y no se genera. Viene versionada aquí, así que basta con clonar. En cambio
`data/processed/` se crea sola y su contenido queda fuera del control de versiones, porque se
regenera. Los modelos de `models/` **sí** se versionan pese a ser binarios: el servicio que aloja
el tablero clona este repositorio y necesita encontrarlos ya entrenados.

## Tablero interactivo

```bash
streamlit run dashboard/app.py
```

Se abre en <http://localhost:8501>. El arranque tarda unos tres segundos, que es lo que cuesta
reconstruir las 8465 horas y cargar los modelos; a partir de ahí queda en caché. El comando se
lanza **desde la raíz del repositorio**, no desde `dashboard/`, porque es así como lo ejecuta el
servicio de despliegue.

Si el entorno no es el de los cuadernos, bastan las siete dependencias del tablero:

```bash
pip install -r dashboard/requirements.txt
```

### Despliegue

La aplicación está pensada para Streamlit Community Cloud, que toma el repositorio público y
redespliega con cada `push`. Al crearla hay que indicar `dashboard/app.py` como archivo de entrada
y **Python 3.11** en las opciones avanzadas, la versión con la que se serializaron los modelos.

Hay dos archivos de dependencias y no es un descuido. `requirements.txt`, en la raíz, reproduce los
cuadernos e incluye torch y shap. `dashboard/requirements.txt` contiene solo lo que la aplicación
necesita, que no entrena nada. Streamlit Community Cloud busca el archivo primero junto al archivo
de entrada y después en la raíz, de modo que usa el segundo; la configuración, en cambio, solo la
lee desde la raíz, y por eso `.streamlit/config.toml` está ahí y no dentro de `dashboard/`.
