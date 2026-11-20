# Métricas de los modelos

Partición estratificada 80/20, semilla 42; CV estratificada de 5 pliegues solo sobre entrenamiento; selección por f1_macro.

## Al terminar el 1.er semestre (principal)

| Escenario | Modelo | CV_F1_macro | CV_F1_macro_std | Test_Accuracy | Test_Balanced_accuracy | Test_F1_macro | Mejor_por_CV |
|---|---|---|---|---|---|---|---|
| sem1 | Baseline (clase mayoritaria) | 0.2220 | 0.0002 | 0.4994 | 0.3333 | 0.2221 | False |
| sem1 | Regresion Logistica | 0.6878 | 0.0180 | 0.7051 | 0.6778 | 0.6677 | False |
| sem1 | Arbol de Decision | 0.6397 | 0.0099 | 0.6463 | 0.6324 | 0.6187 | False |
| sem1 | Random Forest | 0.6993 | 0.0265 | 0.7277 | 0.6735 | 0.6725 | True |
| sem1 | SVM | 0.6899 | 0.0144 | 0.7107 | 0.6844 | 0.6727 | False |

Mejor modelo por F1 macro en CV: **Random Forest**

## Al terminar el 2.º semestre (referencia)

| Escenario | Modelo | CV_F1_macro | CV_F1_macro_std | Test_Accuracy | Test_Balanced_accuracy | Test_F1_macro | Mejor_por_CV |
|---|---|---|---|---|---|---|---|
| sem2 | Baseline (clase mayoritaria) | 0.2220 | 0.0002 | 0.4994 | 0.3333 | 0.2221 | False |
| sem2 | Regresion Logistica | 0.7172 | 0.0207 | 0.7288 | 0.7057 | 0.6927 | False |
| sem2 | Arbol de Decision | 0.6816 | 0.0132 | 0.7062 | 0.6747 | 0.6658 | False |
| sem2 | Random Forest | 0.7261 | 0.0206 | 0.7729 | 0.7306 | 0.7280 | True |
| sem2 | SVM | 0.7243 | 0.0175 | 0.7333 | 0.7064 | 0.6957 | False |

Mejor modelo por F1 macro en CV: **Random Forest**
