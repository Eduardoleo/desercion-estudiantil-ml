"""Pruebas mínimas de reproducibilidad y de ausencia de fuga temporal.

Ejecutar desde la raíz del repositorio:
    python -m unittest discover -s tests -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_SCRIPTS = RAIZ / "Logica Python"
sys.path.insert(0, str(CARPETA_SCRIPTS))

import Leer_dataset  # noqa: E402
import Predecir_desercion as pd_mod  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


def ejecutar(script: str, *args: str) -> subprocess.CompletedProcess:
    # El hijo escribe en UTF-8 sin importar la consola (Git Bash usa cp1252).
    entorno = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(CARPETA_SCRIPTS / script), *args],
                          capture_output=True, text=True, encoding='utf-8', errors='replace',
                          env=entorno)


class TestArchivoFaltante(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inexistente = str(Path(self.tmp.name) / "no_existe.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_leer_dataset_sale_con_error(self):
        resultado = ejecutar("Leer_dataset.py", "--entrada", self.inexistente,
                             "--salida", str(Path(self.tmp.name) / "salida.csv"))
        self.assertNotEqual(resultado.returncode, 0)
        self.assertIn("no se encontró", resultado.stderr)

    def test_predecir_sale_con_error(self):
        resultado = ejecutar("Predecir_desercion.py", "--entrada", self.inexistente,
                             "--resultados", self.tmp.name)
        self.assertNotEqual(resultado.returncode, 0)

    def test_graficos_sale_con_error(self):
        resultado = ejecutar("Generar_Graficos.py", "--resultados", self.tmp.name)
        self.assertNotEqual(resultado.returncode, 0)


class TestHorizontePrimerSemestre(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = Leer_dataset.cargar_dataset(Leer_dataset.RUTA_ENTRADA)
        cls.df = Leer_dataset.limpiar_dataset(original)

    def test_dataset_contiene_columnas_2do_semestre(self):
        # Garantiza que la prueba siguiente no pase por accidente.
        self.assertTrue(any(pd_mod.es_columna_2do_semestre(c) for c in self.df.columns))

    def test_escenario_principal_excluye_2do_semestre(self):
        X, _ = pd_mod.seleccionar_variables(self.df, pd_mod.ESCENARIO_PRINCIPAL)
        prohibidas = [c for c in X.columns if '2nd sem' in c
                      or c in ('Total_Aprobadas_Sem2', 'Total_Aprobadas')]
        self.assertEqual(prohibidas, [])
        self.assertIn('Curricular units 1st sem (approved)', X.columns)

    def test_pipeline_principal_no_usa_2do_semestre(self):
        muestra = self.df.sample(600, random_state=0)
        X, y = pd_mod.seleccionar_variables(muestra, pd_mod.ESCENARIO_PRINCIPAL)
        X = pd_mod.preparar_categoricas(X)
        pipeline = pd_mod.construir_pipeline(X, LogisticRegression(max_iter=2000))
        pipeline.fit(X, y)
        entrada = list(pipeline.feature_names_in_)
        salida = list(pipeline.named_steps['prep'].get_feature_names_out())
        self.assertFalse(any('2nd sem' in c for c in entrada + salida))
        derivadas = ('Total_Aprobadas_Sem2', 'Total_Aprobadas')
        self.assertFalse(any(c in derivadas for c in entrada))
        self.assertFalse(any(c.split('__', 1)[-1] in derivadas for c in salida))

    def test_escenario_referencia_si_incluye_2do_semestre(self):
        X, _ = pd_mod.seleccionar_variables(self.df, 'sem2')
        self.assertTrue(any('2nd sem' in c for c in X.columns))


if __name__ == "__main__":
    unittest.main()
