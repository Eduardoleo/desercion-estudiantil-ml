# ==============================
# EJEMPLO DE CALCULO DE METRICAS
# Para mostrar en presentación
# ==============================
"""
Este script demuestra cómo se calculan las métricas
de manera automática para cada modelo.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report
)
from sklearn.model_selection import cross_val_score
import time

# ==============================
# EJEMPLO SIMPLIFICADO
# ==============================

def calcular_metricas_modelo(modelo, X_train, X_test, y_train, y_test, nombre_modelo):
    """
    Calcula todas las métricas para un modelo dado.
    Esta función se reutiliza para cada algoritmo.
    """
    print(f"\n{'='*70}")
    print(f"CALCULANDO METRICAS PARA: {nombre_modelo}")
    print(f"{'='*70}")
    
    # 1. MEDIR TIEMPO DE ENTRENAMIENTO
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio
    
    # 2. MEDIR TIEMPO DE PREDICCION
    inicio = time.time()
    y_pred = modelo.predict(X_test)
    tiempo_prediccion = time.time() - inicio
    
    # 3. CALCULAR METRICAS PRINCIPALES
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # 4. CROSS-VALIDATION (5-FOLD)
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # 5. ALMACENAR RESULTADOS
    metricas = {
        'Modelo': nombre_modelo,
        'Accuracy': accuracy * 100,          # Métrica 1
        'Precision': precision * 100,        # Métrica 2
        'Recall': recall * 100,              # Métrica 3
        'F1_Score': f1 * 100,                # Métrica 4
        'CV_Mean': cv_mean * 100,            # Métrica 5
        'CV_Std': cv_std * 100,              # Métrica 6
        'Tiempo_Entrenamiento': tiempo_entrenamiento,  # Métrica 7
        'Tiempo_Prediccion': tiempo_prediccion         # Métrica 8
    }
    
    # 6. MOSTRAR RESULTADOS
    print(f"\n Métricas calculadas:")
    print(f"   Accuracy:        {metricas['Accuracy']:.2f}%")
    print(f"   Precision:       {metricas['Precision']:.2f}%")
    print(f"   Recall:          {metricas['Recall']:.2f}%")
    print(f"   F1-Score:        {metricas['F1_Score']:.2f}%")
    print(f"   Cross-Val:       {metricas['CV_Mean']:.2f}% (±{metricas['CV_Std']:.2f}%)")
    print(f"   T. Entrenamiento: {metricas['Tiempo_Entrenamiento']:.4f} seg")
    print(f"   T. Predicción:   {metricas['Tiempo_Prediccion']:.4f} seg")
    
    # 7. REPORTE DETALLADO POR CLASE
    print(f"\n📊 Reporte por clase:")
    print(classification_report(y_test, y_pred))
    
    return metricas


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("ESTE ARCHIVO ES PARA REFERENCIA")
    print("El código real está en: Predecir_desercion.py")
    print("="*70)
