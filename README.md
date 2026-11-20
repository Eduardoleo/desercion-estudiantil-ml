# Deserción estudiantil: predicción temprana con machine learning

**Caso de estudio:** [en español](https://eduardoleo.github.io/proyectos/desercion-estudiantil/) · [in English](https://eduardoleo.github.io/en/projects/desercion-estudiantil/)

Clasificación del estado académico final de estudiantes universitarios
(**abandono**, **sigue inscrito** o **se gradúa**) a partir de datos de
inscripción y del primer semestre, con una metodología de evaluación sin fuga
de datos. Incluye un dashboard de Power BI sobre el dataset limpio y las
predicciones.

**Stack:** Python 3.13 · pandas · scikit-learn · matplotlib · Power BI

## Problema

Detectar a tiempo a quienes tienen riesgo de abandonar permite intervenir
(tutorías, apoyo financiero) antes de que sea tarde. La pregunta práctica no es
"¿se puede predecir la deserción?", sino **"¿qué tan bien se puede predecir con
la información disponible en el momento de actuar?"**.

## Dataset

- **Fuente:** UCI Machine Learning Repository, *Predict Students' Dropout and
  Academic Success*.
- **Cita:** Realinho, V., Vieira Martins, M., Machado, J. y Baptista, L. (2021).
  *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine
  Learning Repository. <https://doi.org/10.24432/C5MC89>
- **Licencia del dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- 4424 estudiantes, 36 variables y la clase `Target` (Graduate 49,9 %,
  Dropout 32,1 %, Enrolled 17,9 %). Sin duplicados. Los atributos están
  codificados y seudonimizados en origen; no hay datos personales identificables.
- Archivo en `Logica Python/Archivos/Predict Students' Dropout and Academic Success.csv`.

## Solución y metodología

| Paso | Script | Qué hace |
|---|---|---|
| 1 | `Leer_dataset.py` | Limpia el CSV original, traduce columnas, convierte códigos binarios a texto y crea columnas calculadas. |
| 2 | `Predecir_desercion.py` | Entrena y evalúa 4 clasificadores y un baseline en dos horizontes de predicción. |
| 3 | `Generar_Graficos.py` | Genera los gráficos a partir de los resultados. |

`Ejemplo_Calculo_Metricas.py` es material de referencia para la presentación;
no forma parte del flujo.

### Horizonte de predicción

- **Escenario principal: al terminar el 1.er semestre.** Solo usa lo que se
  conoce en ese momento (inscripción, datos socioeconómicos, indicadores
  macroeconómicos y rendimiento del 1.er semestre). Se excluyen las 6 columnas
  `Curricular units 2nd sem (*)` y sus derivadas (`Total_Aprobadas_Sem2`,
  `Total_Aprobadas`): quedan 32 variables.
- **Escenario de referencia: al terminar el 2.º semestre** (40 variables). Se
  reporta para medir cuánto aporta esperar un semestre más; no es el resultado
  principal.

### Evaluación

1. Partición **estratificada 80/20** (semilla 42). El conjunto de prueba se
   aparta y se usa **una sola vez** al final.
2. Preprocesamiento **dentro de un `Pipeline`**: one-hot para las categóricas
   (incluidos los códigos nominales como curso u ocupación de los padres) y
   `StandardScaler` para las numéricas. Así el escalado se ajusta solo con
   los datos de cada pliegue de entrenamiento.
3. Hiperparámetros con **`GridSearchCV` y `StratifiedKFold` (5 pliegues)
   solo sobre entrenamiento**, optimizando **F1 macro**, porque las clases
   están desbalanceadas y la clase minoritaria (Enrolled) es la más difícil.
4. **Baseline** `DummyClassifier` (siempre predice la clase mayoritaria).
5. El **mejor modelo se elige por F1 macro medio en validación cruzada**, no
   por el resultado en prueba. Ese modelo genera la columna `Prediccion_ML`.

## Resultados

Valores obtenidos al ejecutar el flujo con las versiones de `requirements.txt`
(conjunto de prueba: 885 estudiantes). Tabla completa en
[`Logica Python/Resultados/metricas_modelos.md`](Logica%20Python/Resultados/metricas_modelos.md);
métricas por clase en `metricas_por_clase.csv`.

**Escenario principal (al terminar el 1.er semestre)**

| Modelo | F1 macro (CV) | Accuracy (prueba) | Balanced accuracy (prueba) | F1 macro (prueba) |
|---|---|---|---|---|
| Baseline (clase mayoritaria) | 0,222 | 0,499 | 0,333 | 0,222 |
| Regresión logística | 0,688 ± 0,018 | 0,705 | 0,678 | 0,668 |
| Árbol de decisión | 0,640 ± 0,010 | 0,646 | 0,632 | 0,619 |
| **Random Forest (elegido)** | **0,699 ± 0,027** | **0,728** | **0,674** | **0,673** |
| SVM (RBF) | 0,690 ± 0,014 | 0,711 | 0,684 | 0,673 |

**Escenario de referencia (al terminar el 2.º semestre)**

| Modelo | F1 macro (CV) | Accuracy (prueba) | Balanced accuracy (prueba) | F1 macro (prueba) |
|---|---|---|---|---|
| Baseline (clase mayoritaria) | 0,222 | 0,499 | 0,333 | 0,222 |
| Regresión logística | 0,717 ± 0,021 | 0,729 | 0,706 | 0,693 |
| Árbol de decisión | 0,682 ± 0,013 | 0,706 | 0,675 | 0,666 |
| **Random Forest (elegido)** | **0,726 ± 0,021** | **0,773** | **0,731** | **0,728** |
| SVM (RBF) | 0,724 ± 0,018 | 0,733 | 0,706 | 0,696 |

Lectura:

- Todos los modelos superan con claridad al baseline (F1 macro 0,22).
- Random Forest tiene el mejor F1 macro en CV en ambos escenarios. En el
  principal, la diferencia con SVM y regresión logística (≈ 0,01) es menor
  que la desviación entre pliegues: los tres son prácticamente equivalentes,
  y la regresión logística es una alternativa más interpretable.
- Con el modelo elegido, el recall en prueba del escenario principal es 0,69
  para Dropout, 0,50 para Enrolled y 0,84 para Graduate. "Sigue inscrito" es
  la clase que más se confunde.
- Esperar al 2.º semestre sube el F1 macro de prueba de 0,673 a 0,728, pero la
  alerta llega un semestre más tarde.

**Antes de la corrección:** la versión anterior reportaba accuracy 0,7955 con
Random Forest. Ese valor estaba inflado: los hiperparámetros se elegían con el
conjunto de prueba, se usaban datos del 2.º semestre sin definir el momento de
la predicción y no había baseline ni métricas por clase.

## Capturas

Gráficos generados por `Generar_Graficos.py` en `Logica Python/Resultados/graficos/`:

- Matriz de confusión del mejor modelo (escenario principal, prueba):
  ![Matriz de confusión](Logica%20Python/Resultados/graficos/1_matriz_confusion_mejor_modelo.png)
- Comparación de modelos por F1 macro (CV y prueba, ambos escenarios):
  ![Comparación F1 macro](Logica%20Python/Resultados/graficos/2_comparacion_f1_macro.png)
- Importancia por permutación de las variables del mejor modelo:
  ![Importancia de variables](Logica%20Python/Resultados/graficos/3_importancia_variables.png)

## Cómo ejecutarlo

Requisitos: Windows con PowerShell y Python 3.13.

```powershell
cd C:\ruta\al\repositorio
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd "Logica Python"
python Leer_dataset.py         # 1. limpia -> Resultados\datos\
python Predecir_desercion.py   # 2. entrena y evalúa (unos segundos por modelo)
python Generar_Graficos.py     # 3. gráficos -> Resultados\graficos\
cd ..
python -m unittest discover -s tests -v
```

Cada script acepta rutas por argumento (`--help`) y termina con código 1 si
falta un archivo de entrada.

Salidas en `Logica Python/Resultados/`:

| Archivo | Contenido |
|---|---|
| `metricas_modelos.csv` / `.md` | Métricas de CV y prueba e hiperparámetros por modelo y escenario |
| `metricas_por_clase.csv` | Precisión, recall, F1 y soporte por clase |
| `matrices_confusion.csv` | Matrices de confusión en formato largo |
| `importancia_variables.csv` | Importancia por permutación del mejor modelo |
| `resumen.json` | Mejor modelo, variables excluidas, semilla y versiones |
| `graficos/*.png` | Gráficos |
| `datos/` (no versionada) | CSV limpio y CSV con predicciones para Power BI |

El CSV de predicciones incluye la columna `Conjunto` (Entrenamiento/Prueba):
las predicciones sobre filas de entrenamiento no miden desempeño.

> **Nota sobre Power BI:** el tablero `.pbix` debe actualizar su origen de
> datos a `Logica Python/Resultados/datos/`, donde quedan los CSV que generan
> los scripts anteriores (esa carpeta no se versiona: hay que ejecutar los
> scripts antes de abrir el tablero). Este cambio de origen lo hará el dueño
> del repositorio en Power BI Desktop; mientras tanto, el `.pbix` puede seguir
> apuntando a rutas antiguas.

## Proyecto grupal

Proyecto de un equipo de 3 estudiantes de la UIA (Costa Rica), curso
Inteligencia Artificial Aplicada, 2025.

**Mi contribución:** desarrollo principal: limpieza y preparación de datos,
modelos en Python y tablero de Power BI. Proyecto académico en equipo; los
compañeros contribuyeron con correcciones puntuales.

La corrección metodológica, la reproducibilidad y las pruebas de este
repositorio se hicieron después de la entrega, sobre una copia de trabajo.

Los documentos del grupo (`*.docx`, `*.pdf`, `*.pptx`) y el archivo
`Dashboard Proyecto Grupo 3.pbix` contienen nombres de integrantes y del
profesor: **no forman parte de la publicación pública** y se excluyen del
repositorio público. Por eso este README muestra los gráficos que genera
Python y no capturas del tablero. Quien tenga el `.pbix` debe dirigir sus
orígenes de datos a `Resultados/datos/` (antes apuntaban a `Archivos/`).

## Lo que aprendí

- Una métrica alta no sirve si no se define **cuándo** se hace la predicción.
  Con datos del 2.º semestre el modelo "predice" con información que todavía
  no existiría en el momento de intervenir.
- Elegir hiperparámetros con el conjunto de prueba lo convierte en parte del
  entrenamiento. La selección debe hacerse con validación cruzada solo en
  entrenamiento, y el test se usa una vez.
- El preprocesamiento (escalado, codificación) va dentro del `Pipeline` para
  que cada pliegue se ajuste solo con sus propios datos.
- Con clases desbalanceadas, accuracy engaña: el baseline que siempre dice
  "Graduate" acierta el 50 %. F1 macro, balanced accuracy y las métricas por
  clase muestran mejor el desempeño real.
- Un baseline y la desviación entre pliegues ponen en contexto las diferencias
  entre modelos: 0,01 de F1 macro no siempre es una mejora real.

## Limitaciones

- Es un dataset de una sola institución (Portugal); los resultados no se
  trasladan sin más a otra universidad.
- No se sabe con certeza en qué momento se registran algunas variables, como
  `Matricula_Al_Dia` o `Deudor`. Si se actualizan después del 1.er semestre,
  podría quedar una fuga residual.
- La importancia de variables se calcula en entrenamiento para no tocar el
  conjunto de prueba; con Random Forest puede sobrestimar algunas variables.
- La rejilla de hiperparámetros es acotada. Hay una sola partición de prueba,
  sin intervalos de confianza para las métricas de prueba.
- Las variables sensibles (género, edad, situación económica) influyen en el
  modelo. Antes de cualquier uso real haría falta revisar sesgos y equidad;
  esto es un ejercicio académico, no una herramienta de decisión.

## Licencia

- **Código:** © 2025–2026 Eduardo León Hernández. Todos los derechos reservados
  (ver [LICENSE](LICENSE)). Se publica como muestra de trabajo: se puede ver y
  evaluar, pero no copiar, modificar, distribuir ni reutilizar sin permiso escrito.
- **Dataset:** CC BY 4.0, de sus autores (ver la cita).
