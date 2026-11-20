# ==============================
# VISUALIZACIONES DE METRICAS ML
# ==============================
"""
Genera los gráficos a partir de los resultados de Predecir_desercion.py:

1. Matriz de confusión del mejor modelo (escenario principal, conjunto de prueba).
2. Comparación de modelos por F1 macro (CV en entrenamiento vs. prueba), ambos escenarios.
3. Importancia por permutación de las variables del mejor modelo (escenario principal).

Uso:
    python Generar_Graficos.py [--resultados CARPETA]

Sale con código 1 si faltan los archivos de resultados.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # sin ventana: solo archivos PNG
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
CARPETA_RESULTADOS = BASE_DIR / "Resultados"
CLASES = ['Dropout', 'Enrolled', 'Graduate']
ARCHIVOS_REQUERIDOS = ['metricas_modelos.csv', 'matrices_confusion.csv',
                       'importancia_variables.csv', 'resumen.json']

# Paleta sobria (validada para daltonismo en pares adyacentes) y tinta de texto.
SERIE_1 = '#2a78d6'  # azul
SERIE_2 = '#eb6834'  # naranja
TINTA = '#0b0b0b'
TINTA_SECUNDARIA = '#52514e'
REJILLA = '#e4e3df'
FONDO = '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': FONDO, 'axes.facecolor': FONDO, 'savefig.facecolor': FONDO,
    'axes.edgecolor': REJILLA, 'axes.labelcolor': TINTA_SECUNDARIA,
    'xtick.color': TINTA_SECUNDARIA, 'ytick.color': TINTA_SECUNDARIA,
    'text.color': TINTA, 'font.size': 10, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
})


def cargar_resultados(carpeta: Path):
    faltantes = [n for n in ARCHIVOS_REQUERIDOS if not (carpeta / n).is_file()]
    if faltantes:
        raise FileNotFoundError(', '.join(faltantes))
    metricas = pd.read_csv(carpeta / 'metricas_modelos.csv')
    matrices = pd.read_csv(carpeta / 'matrices_confusion.csv')
    importancia = pd.read_csv(carpeta / 'importancia_variables.csv')
    resumen = json.loads((carpeta / 'resumen.json').read_text(encoding='utf-8'))
    return metricas, matrices, importancia, resumen


def grafico_matriz_confusion(matrices, resumen, destino: Path) -> Path:
    escenario = resumen['escenario_principal']
    info = resumen['escenarios'][escenario]
    modelo = info['mejor_modelo']
    datos = matrices[(matrices['Escenario'] == escenario) & (matrices['Modelo'] == modelo)]
    matriz = (datos.pivot(index='Real', columns='Predicho', values='Cantidad')
              .reindex(index=CLASES, columns=CLASES).to_numpy())
    por_fila = matriz / matriz.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    imagen = ax.imshow(por_fila, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            color = 'white' if por_fila[i, j] > 0.55 else TINTA
            ax.text(j, i, f"{matriz[i, j]}\n{por_fila[i, j]:.0%}", ha='center', va='center',
                    color=color, fontsize=11)
    ax.set_xticks(range(len(CLASES)), CLASES)
    ax.set_yticks(range(len(CLASES)), CLASES)
    ax.set_xlabel('Clase predicha')
    ax.set_ylabel('Clase real')
    ax.spines[:].set_visible(False)
    ax.set_title(f"Matriz de confusión — {modelo}\n{info['etiqueta']}, conjunto de prueba")
    barra = fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
    barra.set_label('Proporción de la clase real (recall)')
    barra.outline.set_visible(False)
    fig.tight_layout()
    ruta = destino / '1_matriz_confusion_mejor_modelo.png'
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ruta


def grafico_comparacion_f1(metricas, resumen, destino: Path) -> Path:
    escenarios = [e for e in resumen['escenarios']]
    fig, ejes = plt.subplots(1, len(escenarios), figsize=(6.2 * len(escenarios), 4.6),
                             sharex=True, squeeze=False)
    for ax, escenario in zip(ejes[0], escenarios):
        datos = metricas[metricas['Escenario'] == escenario].reset_index(drop=True)
        y = np.arange(len(datos))
        alto = 0.36
        ax.barh(y - alto / 2 - 0.01, datos['CV_F1_macro'], alto, xerr=datos['CV_F1_macro_std'],
                color=SERIE_1, label='CV (entrenamiento, media ± desv.)',
                error_kw={'ecolor': TINTA_SECUNDARIA, 'lw': 1, 'capsize': 3})
        ax.barh(y + alto / 2 + 0.01, datos['Test_F1_macro'], alto, color=SERIE_2,
                label='Prueba (una sola evaluación)')
        for i, fila in datos.iterrows():
            ax.text(fila['Test_F1_macro'] + 0.01, i + alto / 2 + 0.01,
                    f"{fila['Test_F1_macro']:.3f}", va='center', fontsize=8,
                    color=TINTA_SECUNDARIA)
        etiquetas = [f"{m}  (mejor)" if mejor else m
                     for m, mejor in zip(datos['Modelo'], datos['Mejor_por_CV'])]
        ax.set_yticks(y, etiquetas)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(axis='x', color=REJILLA, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel('F1 macro')
        ax.set_title(resumen['escenarios'][escenario]['etiqueta'])
    manejadores, textos = ejes[0][0].get_legend_handles_labels()
    fig.suptitle('Comparación de modelos por F1 macro (selección por CV, no por prueba)',
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(manejadores, textos, loc='lower center', ncol=2, frameon=False)
    ruta = destino / '2_comparacion_f1_macro.png'
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ruta


def grafico_importancia(importancia, resumen, destino: Path, n: int = 15) -> Path:
    escenario = resumen['escenario_principal']
    datos = (importancia[importancia['Escenario'] == escenario]
             .sort_values('Importancia_media', ascending=False).head(n))
    modelo = datos['Modelo'].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(datos) + 1.4))
    ax.barh(datos['Variable'], datos['Importancia_media'], xerr=datos['Importancia_std'],
            color=SERIE_1, error_kw={'ecolor': TINTA_SECUNDARIA, 'lw': 1, 'capsize': 2})
    ax.invert_yaxis()
    ax.grid(axis='x', color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel('Caída media de F1 macro al permutar la variable (entrenamiento)')
    ax.set_title(f"Importancia por permutación — {modelo}\n"
                 f"{resumen['escenarios'][escenario]['etiqueta']}, {n} variables principales")
    fig.tight_layout()
    ruta = destino / '3_importancia_variables.png'
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ruta


def parsear_argumentos(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--resultados', type=Path, default=CARPETA_RESULTADOS,
                        help='Carpeta generada por Predecir_desercion.py')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parsear_argumentos(argv)
    try:
        metricas, matrices, importancia, resumen = cargar_resultados(args.resultados)
    except FileNotFoundError as error:
        print(f"ERROR: faltan resultados en {args.resultados}: {error}", file=sys.stderr)
        print("Ejecuta primero Predecir_desercion.py.", file=sys.stderr)
        return 1

    destino = args.resultados / 'graficos'
    destino.mkdir(parents=True, exist_ok=True)
    rutas = [
        grafico_matriz_confusion(matrices, resumen, destino),
        grafico_comparacion_f1(metricas, resumen, destino),
        grafico_importancia(importancia, resumen, destino),
    ]
    print("Gráficos generados:")
    for ruta in rutas:
        print(f"  - {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
