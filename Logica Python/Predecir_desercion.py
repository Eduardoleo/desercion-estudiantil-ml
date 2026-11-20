# ==============================
# PREDICCION DE DESERCION ESTUDIANTIL
# ==============================
"""
Entrena y compara clasificadores para predecir el estado académico final de un
estudiante (Dropout / Enrolled / Graduate) con una metodología sin fuga de datos.

Horizonte de predicción
-----------------------
- Escenario principal "sem1": la predicción se hace AL TERMINAR EL 1.er SEMESTRE.
  Solo se usan variables conocidas en ese momento: datos de inscripción,
  socioeconómicos, macroeconómicos y del 1.er semestre. Se excluyen todas las
  columnas "Curricular units 2nd sem (*)" y las derivadas de ellas
  (Total_Aprobadas_Sem2, Total_Aprobadas).
- Escenario secundario "sem2": AL TERMINAR EL 2.º SEMESTRE (todas las variables).
  Se reporta solo como referencia; su desempeño es mayor porque usa más
  información, pero la alerta llega un semestre más tarde.

Metodología
-----------
1. Partición estratificada 80/20 (semilla fija). El conjunto de prueba se aparta
   y se usa UNA sola vez al final.
2. Preprocesamiento (one-hot de categóricas y StandardScaler de numéricas)
   dentro de un sklearn Pipeline, para que se ajuste solo con los datos de cada
   pliegue de entrenamiento.
3. Hiperparámetros con GridSearchCV + StratifiedKFold (5 pliegues) SOLO sobre
   entrenamiento, optimizando F1 macro (las clases están desbalanceadas).
4. Baseline: DummyClassifier (clase mayoritaria).
5. El "mejor modelo" se elige por F1 macro medio en validación cruzada, no por
   el resultado en prueba.

Uso:
    python Predecir_desercion.py [--entrada CSV_LIMPIO] [--resultados CARPETA]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ==============================
# CONFIGURACION
# ==============================
BASE_DIR = Path(__file__).resolve().parent
CARPETA_RESULTADOS = BASE_DIR / "Resultados"
RUTA_DATOS = CARPETA_RESULTADOS / "datos" / "Predict Students Dropout And Academic Success Update.csv"

COLUMNA_TARGET = 'Estado_Academico'
CLASES = ['Dropout', 'Enrolled', 'Graduate']
TAMANO_PRUEBA = 0.2
SEMILLA = 42
PLIEGUES_CV = 5
METRICA_SELECCION = 'f1_macro'

ESCENARIO_PRINCIPAL = 'sem1'
ESCENARIOS = {
    'sem1': {'etiqueta': 'Al terminar el 1.er semestre (principal)', 'incluye_2do_sem': False},
    'sem2': {'etiqueta': 'Al terminar el 2.º semestre (referencia)', 'incluye_2do_sem': True},
}

# Columnas derivadas del 2.º semestre creadas en Leer_dataset.py
DERIVADAS_2DO_SEM = ['Total_Aprobadas_Sem2', 'Total_Aprobadas']

# Variables categóricas: texto (binarias o nominales) y códigos nominales del
# dataset original que no tienen orden y por eso se codifican one-hot.
CATEGORICAS = [
    'Genero', 'Estado_Civil', 'Asistencia_Diurna', 'Desplazado',
    'Necesidades_Especiales', 'Deudor', 'Matricula_Al_Dia', 'Becado',
    'Internacional', 'Modo_Aplicacion', 'Curso', 'Calificacion_Previa',
    'Nacionalidad', 'Calificacion_Madre', 'Calificacion_Padre',
    'Ocupacion_Madre', 'Ocupacion_Padre',
]

NOMBRE_BASELINE = 'Baseline (clase mayoritaria)'


# ==============================
# DATOS Y VARIABLES
# ==============================
def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Lee el CSV limpio generado por Leer_dataset.py."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(ruta)
    df = pd.read_csv(ruta)
    if COLUMNA_TARGET not in df.columns:
        raise ValueError(f"El archivo no contiene la columna '{COLUMNA_TARGET}'")
    return df


def es_columna_2do_semestre(columna: str) -> bool:
    return '2nd sem' in columna or columna in DERIVADAS_2DO_SEM


def seleccionar_variables(df: pd.DataFrame, escenario: str):
    """Devuelve (X, y) con las variables disponibles en el horizonte indicado."""
    if escenario not in ESCENARIOS:
        raise ValueError(f"Escenario desconocido: {escenario}")
    columnas = [c for c in df.columns if c != COLUMNA_TARGET]
    if not ESCENARIOS[escenario]['incluye_2do_sem']:
        columnas = [c for c in columnas if not es_columna_2do_semestre(c)]
    return df[columnas].copy(), df[COLUMNA_TARGET].copy()


def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    categoricas = [c for c in CATEGORICAS if c in X.columns]
    numericas = [c for c in X.columns if c not in categoricas]
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20,
                              sparse_output=False), categoricas),
        ('num', StandardScaler(), numericas),
    ])


def definir_modelos() -> dict:
    """Modelo base y rejilla de hiperparámetros (prefijo 'clf__')."""
    return {
        NOMBRE_BASELINE: (DummyClassifier(strategy='most_frequent'), {}),
        'Regresion Logistica': (
            LogisticRegression(max_iter=5000, random_state=SEMILLA),
            {'clf__C': [0.1, 1.0, 10.0], 'clf__class_weight': [None, 'balanced']},
        ),
        'Arbol de Decision': (
            DecisionTreeClassifier(random_state=SEMILLA),
            {'clf__max_depth': [4, 6, 10, None], 'clf__min_samples_leaf': [1, 10, 30],
             'clf__class_weight': [None, 'balanced']},
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=300, random_state=SEMILLA, n_jobs=1),
            {'clf__max_depth': [10, 20, None], 'clf__min_samples_leaf': [1, 3],
             'clf__class_weight': [None, 'balanced']},
        ),
        'SVM': (
            SVC(kernel='rbf', random_state=SEMILLA),
            {'clf__C': [0.5, 1.0, 5.0], 'clf__gamma': ['scale', 0.01],
             'clf__class_weight': [None, 'balanced']},
        ),
    }


def construir_pipeline(X: pd.DataFrame, estimador) -> Pipeline:
    return Pipeline([('prep', construir_preprocesador(X)), ('clf', estimador)])


def preparar_categoricas(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for columna in CATEGORICAS:
        if columna in X.columns:
            X[columna] = X[columna].astype(str)
    return X


# ==============================
# ENTRENAMIENTO Y EVALUACION
# ==============================
def entrenar_escenario(df: pd.DataFrame, escenario: str):
    """Ajusta todos los modelos de un escenario. Devuelve un dict con resultados."""
    X, y = seleccionar_variables(df, escenario)
    X = preparar_categoricas(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAMANO_PRUEBA, random_state=SEMILLA, stratify=y)

    cv = StratifiedKFold(n_splits=PLIEGUES_CV, shuffle=True, random_state=SEMILLA)
    scoring = {'f1_macro': 'f1_macro', 'balanced_accuracy': 'balanced_accuracy',
               'accuracy': 'accuracy'}

    ajustados, filas = {}, []
    for nombre, (estimador, rejilla) in definir_modelos().items():
        print(f"\n[{escenario}] {nombre}: búsqueda en rejilla con CV de {PLIEGUES_CV} pliegues...")
        inicio = time.perf_counter()
        busqueda = GridSearchCV(construir_pipeline(X_train, estimador), rejilla or [{}],
                                scoring=scoring, refit=METRICA_SELECCION, cv=cv, n_jobs=-1)
        busqueda.fit(X_train, y_train)
        duracion = time.perf_counter() - inicio
        i = busqueda.best_index_
        res = busqueda.cv_results_
        ajustados[nombre] = busqueda.best_estimator_
        filas.append({
            'Escenario': escenario,
            'Modelo': nombre,
            'CV_F1_macro': res['mean_test_f1_macro'][i],
            'CV_F1_macro_std': res['std_test_f1_macro'][i],
            'CV_Balanced_accuracy': res['mean_test_balanced_accuracy'][i],
            'CV_Accuracy': res['mean_test_accuracy'][i],
            'Mejores_hiperparametros': json.dumps(
                {k.replace('clf__', ''): v for k, v in busqueda.best_params_.items()}),
        })
        print(f"   F1 macro CV = {res['mean_test_f1_macro'][i]:.4f} "
              f"(± {res['std_test_f1_macro'][i]:.4f}) | {busqueda.best_params_} | {duracion:.1f} s")

    metricas = pd.DataFrame(filas)
    candidatos = metricas[metricas['Modelo'] != NOMBRE_BASELINE]
    mejor = candidatos.loc[candidatos['CV_F1_macro'].idxmax(), 'Modelo']
    metricas['Mejor_por_CV'] = metricas['Modelo'] == mejor

    # Evaluación final: el conjunto de prueba se usa una sola vez por modelo,
    # después de haber fijado hiperparámetros y elegido el mejor modelo.
    por_clase, matrices = [], []
    for idx, nombre in metricas['Modelo'].items():
        y_pred = ajustados[nombre].predict(X_test)
        metricas.loc[idx, 'Test_Accuracy'] = accuracy_score(y_test, y_pred)
        metricas.loc[idx, 'Test_Balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
        metricas.loc[idx, 'Test_F1_macro'] = f1_score(y_test, y_pred, average='macro',
                                                      zero_division=0)
        reporte = classification_report(y_test, y_pred, labels=CLASES, output_dict=True,
                                        zero_division=0)
        for clase in CLASES:
            por_clase.append({'Escenario': escenario, 'Modelo': nombre, 'Clase': clase,
                              'Precision': reporte[clase]['precision'],
                              'Recall': reporte[clase]['recall'],
                              'F1': reporte[clase]['f1-score'],
                              'Soporte': int(reporte[clase]['support'])})
        matriz = confusion_matrix(y_test, y_pred, labels=CLASES)
        for i_real, real in enumerate(CLASES):
            for i_pred, pred in enumerate(CLASES):
                matrices.append({'Escenario': escenario, 'Modelo': nombre, 'Real': real,
                                 'Predicho': pred, 'Cantidad': int(matriz[i_real, i_pred])})
        print(f"\n[{escenario}] {nombre} — conjunto de prueba")
        print(classification_report(y_test, y_pred, labels=CLASES, digits=4, zero_division=0))
        print("Matriz de confusión (filas = real, columnas = predicho):", CLASES)
        print(matriz)

    return {
        'escenario': escenario,
        'metricas': metricas,
        'por_clase': pd.DataFrame(por_clase),
        'matrices': pd.DataFrame(matrices),
        'modelos': ajustados,
        'mejor': mejor,
        'X': X, 'X_train': X_train, 'y_train': y_train, 'indices_prueba': X_test.index,
        'variables': list(X.columns),
    }


def importancia_variables(resultado: dict) -> pd.DataFrame:
    """Importancia por permutación del mejor modelo, calculada sobre ENTRENAMIENTO
    (no se toca el conjunto de prueba). Mide la caída de F1 macro."""
    modelo = resultado['modelos'][resultado['mejor']]
    imp = permutation_importance(modelo, resultado['X_train'], resultado['y_train'],
                                 scoring=METRICA_SELECCION, n_repeats=5,
                                 random_state=SEMILLA, n_jobs=-1)
    tabla = pd.DataFrame({
        'Escenario': resultado['escenario'],
        'Modelo': resultado['mejor'],
        'Variable': resultado['X_train'].columns,
        'Importancia_media': imp.importances_mean,
        'Importancia_std': imp.importances_std,
    })
    return tabla.sort_values('Importancia_media', ascending=False).reset_index(drop=True)


# ==============================
# EXPORTACION
# ==============================
def tabla_markdown(df: pd.DataFrame) -> str:
    def formato(valor):
        if isinstance(valor, (float, np.floating)):
            return f"{valor:.4f}"
        return str(valor)
    encabezado = '| ' + ' | '.join(df.columns) + ' |'
    separador = '|' + '|'.join('---' for _ in df.columns) + '|'
    filas = ['| ' + ' | '.join(formato(v) for v in fila) + ' |' for fila in df.itertuples(index=False)]
    return '\n'.join([encabezado, separador, *filas])


def exportar(resultados: list, df: pd.DataFrame, carpeta: Path) -> dict:
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / 'datos').mkdir(parents=True, exist_ok=True)

    metricas = pd.concat([r['metricas'] for r in resultados], ignore_index=True)
    por_clase = pd.concat([r['por_clase'] for r in resultados], ignore_index=True)
    matrices = pd.concat([r['matrices'] for r in resultados], ignore_index=True)
    principal = next(r for r in resultados if r['escenario'] == ESCENARIO_PRINCIPAL)
    importancias = importancia_variables(principal)

    rutas = {
        'metricas': carpeta / 'metricas_modelos.csv',
        'metricas_md': carpeta / 'metricas_modelos.md',
        'por_clase': carpeta / 'metricas_por_clase.csv',
        'matrices': carpeta / 'matrices_confusion.csv',
        'importancia': carpeta / 'importancia_variables.csv',
        'resumen': carpeta / 'resumen.json',
        'predicciones': carpeta / 'datos' / 'Predict Students Dropout And Academic Success ML.csv',
    }
    metricas.to_csv(rutas['metricas'], index=False, encoding='utf-8-sig')
    por_clase.to_csv(rutas['por_clase'], index=False, encoding='utf-8-sig')
    matrices.to_csv(rutas['matrices'], index=False, encoding='utf-8-sig')
    importancias.to_csv(rutas['importancia'], index=False, encoding='utf-8-sig')

    columnas_md = ['Escenario', 'Modelo', 'CV_F1_macro', 'CV_F1_macro_std', 'Test_Accuracy',
                   'Test_Balanced_accuracy', 'Test_F1_macro', 'Mejor_por_CV']
    lineas = ['# Métricas de los modelos', '',
              f'Partición estratificada {int((1 - TAMANO_PRUEBA) * 100)}/{int(TAMANO_PRUEBA * 100)}, '
              f'semilla {SEMILLA}; CV estratificada de {PLIEGUES_CV} pliegues solo sobre '
              f'entrenamiento; selección por {METRICA_SELECCION}.', '']
    for r in resultados:
        lineas += [f"## {ESCENARIOS[r['escenario']]['etiqueta']}", '',
                   tabla_markdown(r['metricas'][columnas_md]), '',
                   f"Mejor modelo por F1 macro en CV: **{r['mejor']}**", '']
    rutas['metricas_md'].write_text('\n'.join(lineas), encoding='utf-8')

    # Dataset con predicciones del escenario principal (para Power BI).
    X = principal['X']
    salida = df.copy()
    salida['Escenario_Prediccion'] = ESCENARIOS[ESCENARIO_PRINCIPAL]['etiqueta']
    salida['Conjunto'] = np.where(salida.index.isin(principal['indices_prueba']),
                                  'Prueba', 'Entrenamiento')
    for nombre, modelo in principal['modelos'].items():
        if nombre == NOMBRE_BASELINE:
            continue
        salida[f"Prediccion_{nombre.replace(' ', '_')}"] = modelo.predict(X)
    salida['Prediccion_ML'] = principal['modelos'][principal['mejor']].predict(X)
    salida['Modelo_Usado'] = principal['mejor']
    salida.to_csv(rutas['predicciones'], index=False, encoding='utf-8-sig')

    resumen = {
        'escenario_principal': ESCENARIO_PRINCIPAL,
        'escenarios': {r['escenario']: {
            'etiqueta': ESCENARIOS[r['escenario']]['etiqueta'],
            'mejor_modelo': r['mejor'],
            'n_variables': len(r['variables']),
            'variables_excluidas': [c for c in df.columns
                                    if c != COLUMNA_TARGET and c not in r['variables']],
        } for r in resultados},
        'criterio_seleccion': f'{METRICA_SELECCION} medio en StratifiedKFold({PLIEGUES_CV}) sobre entrenamiento',
        'semilla': SEMILLA,
        'tamano_prueba': TAMANO_PRUEBA,
        'n_registros': int(len(df)),
        'versiones': {'python': sys.version.split()[0], 'pandas': pd.__version__,
                      'numpy': np.__version__, 'scikit-learn': sklearn.__version__},
    }
    rutas['resumen'].write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
    return rutas


def parsear_argumentos(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entrada', type=Path, default=RUTA_DATOS,
                        help='CSV limpio generado por Leer_dataset.py')
    parser.add_argument('--resultados', type=Path, default=CARPETA_RESULTADOS,
                        help='Carpeta de salida de métricas y predicciones')
    parser.add_argument('--escenarios', nargs='+', default=list(ESCENARIOS),
                        choices=list(ESCENARIOS), help='Escenarios a ejecutar')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parsear_argumentos(argv)
    print("=" * 70)
    print("PREDICCION DE DESERCION ESTUDIANTIL")
    print("=" * 70)
    try:
        df = cargar_datos(args.entrada)
    except FileNotFoundError:
        print(f"ERROR: no se encontró '{args.entrada}'.", file=sys.stderr)
        print("Ejecuta primero 'Leer_dataset.py' para generar el archivo limpio.", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Registros: {len(df)} | Distribución de clases:")
    print(df[COLUMNA_TARGET].value_counts(normalize=True).round(4).to_string())

    escenarios = args.escenarios
    if ESCENARIO_PRINCIPAL not in escenarios:
        escenarios = [ESCENARIO_PRINCIPAL, *escenarios]
    resultados = [entrenar_escenario(df, e) for e in escenarios]
    rutas = exportar(resultados, df, args.resultados)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for r in resultados:
        print(f"\n{ESCENARIOS[r['escenario']]['etiqueta']} — mejor por F1 macro (CV): {r['mejor']}")
        print(r['metricas'][['Modelo', 'CV_F1_macro', 'Test_Accuracy',
                             'Test_Balanced_accuracy', 'Test_F1_macro']]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nArchivos generados:")
    for ruta in rutas.values():
        print(f"  - {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
