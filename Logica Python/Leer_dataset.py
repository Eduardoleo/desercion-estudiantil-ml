# ==============================
# Cargar, limpiar y preparar el dataset (Python y Power BI)
# ==============================
"""
Lee el CSV original de UCI ("Predict Students' Dropout and Academic Success"),
elimina duplicados, traduce nombres de columnas, convierte códigos binarios y
algunos categóricos a texto y crea columnas calculadas.

Uso:
    python Leer_dataset.py [--entrada RUTA_CSV] [--salida RUTA_CSV]

Sale con código 1 si el archivo de entrada no existe o no se puede procesar.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RUTA_ENTRADA = BASE_DIR / "Archivos" / "Predict Students' Dropout and Academic Success.csv"
CARPETA_RESULTADOS = BASE_DIR / "Resultados"
RUTA_SALIDA = CARPETA_RESULTADOS / "datos" / "Predict Students Dropout And Academic Success Update.csv"

NOMBRES_COLUMNAS = {
    'Marital status': 'Estado_Civil',
    'Application mode': 'Modo_Aplicacion',
    'Application order': 'Orden_Aplicacion',
    'Course': 'Curso',
    'Daytime/evening attendance\t': 'Asistencia_Diurna',
    'Daytime/evening attendance': 'Asistencia_Diurna',
    'Previous qualification': 'Calificacion_Previa',
    'Previous qualification (grade)': 'Nota_Previa',
    'Nacionality': 'Nacionalidad',
    "Mother's qualification": 'Calificacion_Madre',
    "Father's qualification": 'Calificacion_Padre',
    "Mother's occupation": 'Ocupacion_Madre',
    "Father's occupation": 'Ocupacion_Padre',
    'Admission grade': 'Nota_Admision',
    'Displaced': 'Desplazado',
    'Educational special needs': 'Necesidades_Especiales',
    'Debtor': 'Deudor',
    'Tuition fees up to date': 'Matricula_Al_Dia',
    'Gender': 'Genero',
    'Scholarship holder': 'Becado',
    'Age at enrollment': 'Edad_Inscripcion',
    'International': 'Internacional',
    'Unemployment rate': 'Tasa_Desempleo',
    'Inflation rate': 'Tasa_Inflacion',
    'GDP': 'PIB',
    'Target': 'Estado_Academico',
}

MAPEOS_TEXTO = {
    'Genero': {0: 'Female', 1: 'Male'},
    'Estado_Civil': {1: 'Soltero', 2: 'Casado', 3: 'Viudo', 4: 'Divorciado',
                     5: 'Union de Hecho', 6: 'Separado'},
    'Asistencia_Diurna': {1: 'Diurna', 0: 'Nocturna'},
}
COLUMNAS_SI_NO = ['Desplazado', 'Necesidades_Especiales', 'Deudor',
                  'Matricula_Al_Dia', 'Becado', 'Internacional']


def cargar_dataset(ruta: Path) -> pd.DataFrame:
    """Lee el CSV original (separado por ';'). Lanza FileNotFoundError si no existe."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(ruta)
    return pd.read_csv(ruta, sep=';')


def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia limpia y normalizada del dataset."""
    df = df.drop_duplicates().copy()
    df.columns = [c.strip() if c.strip() == 'Daytime/evening attendance' else c
                  for c in df.columns]
    df = df.rename(columns=NOMBRES_COLUMNAS)

    for columna, mapa in MAPEOS_TEXTO.items():
        if columna in df.columns:
            df[columna] = df[columna].map(mapa)
    for columna in COLUMNAS_SI_NO:
        if columna in df.columns:
            df[columna] = df[columna].map({1: 'Si', 0: 'No'})

    # Columnas calculadas (las del 2.º semestre se excluyen del escenario
    # principal de predicción; ver Predecir_desercion.py).
    sem1 = 'Curricular units 1st sem (approved)'
    sem2 = 'Curricular units 2nd sem (approved)'
    inscritas1 = 'Curricular units 1st sem (enrolled)'
    if sem1 in df.columns:
        df['Total_Aprobadas_Sem1'] = df[sem1]
    if sem2 in df.columns:
        df['Total_Aprobadas_Sem2'] = df[sem2]
    if sem1 in df.columns and sem2 in df.columns:
        df['Total_Aprobadas'] = df[sem1] + df[sem2]
    if sem1 in df.columns and inscritas1 in df.columns:
        tasa = (df[sem1] / df[inscritas1]).replace([np.inf, -np.inf], np.nan)
        df['Tasa_Aprobacion_Sem1'] = tasa.fillna(0) * 100
    return df


def parsear_argumentos(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entrada', type=Path, default=RUTA_ENTRADA,
                        help='CSV original de UCI (separado por ;)')
    parser.add_argument('--salida', type=Path, default=RUTA_SALIDA,
                        help='CSV limpio de salida')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parsear_argumentos(argv)
    print("=" * 50)
    print("CARGANDO DATASET")
    print("=" * 50)
    try:
        df_original = cargar_dataset(args.entrada)
    except FileNotFoundError:
        print(f"ERROR: no se encontró el archivo de entrada: {args.entrada}", file=sys.stderr)
        print("Descarga el dataset de UCI y colócalo en la carpeta 'Archivos/'.", file=sys.stderr)
        return 1
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        print(f"ERROR: no se pudo leer {args.entrada}: {error}", file=sys.stderr)
        return 1

    print(f"  - Registros: {len(df_original)}")
    print(f"  - Columnas: {len(df_original.columns)}")

    df = limpiar_dataset(df_original)
    print(f"  - Duplicados eliminados: {len(df_original) - len(df)}")
    if 'Estado_Academico' not in df.columns:
        print("ERROR: el CSV no contiene la columna 'Target'.", file=sys.stderr)
        return 1

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False, encoding='utf-8-sig')
    print(f"CSV limpio exportado: {args.salida}")
    print("PROCESO COMPLETADO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
