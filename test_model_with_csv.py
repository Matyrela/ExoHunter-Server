import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import sys
import os

# Add the models directory to the path
sys.path.append('models')

def predecir_con_csv(csv_path='PS_2025.10.02_11.56.01.csv', model_path='modelo_1dcnn.h5'):
    """
    Función para hacer predicciones sobre un CSV usando el modelo entrenado
    """
    print(f"Cargando datos desde: {csv_path}")
    
    # Cargar el modelo entrenado
    model = load_model(model_path)
    print("Modelo cargado exitosamente")
    
    # Cargar el scaler y otros archivos necesarios
    try:
        scaler = joblib.load('scaler.save')
        class_names = np.load('class_names.npy', allow_pickle=True)
        print("Scaler y nombres de clases cargados")
    except Exception as e:
        print(f"Error cargando archivos auxiliares: {e}")
        return
    
    # Leer el CSV
    try:
        df = pd.read_csv(csv_path, comment='#')
        print(f"CSV cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        return
    
    # Preparar los datos de la misma manera que en el entrenamiento
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Si existe discoverymethod, la removemos de las características
    if 'discoverymethod' in df.columns:
        y_true = df['discoverymethod']
        if 'discoverymethod' in numeric_columns:
            numeric_columns.remove('discoverymethod')
        print(f"Columna objetivo encontrada: discoverymethod")
    else:
        y_true = None
        print("No se encontró columna objetivo")
    
    X = df[numeric_columns]
    print(f"Características numéricas: {len(numeric_columns)}")
    
    # Limpiar datos (mismo proceso que en entrenamiento)
    threshold = 0.7
    columns_to_keep = [col for col in X.columns if X[col].isnull().sum() / len(X) < threshold]
    X = X[columns_to_keep]
    
    if y_true is not None:
        row_threshold = len(X.columns) * 0.5
        rows_to_keep = X.isnull().sum(axis=1) < row_threshold
        X = X[rows_to_keep]
        y_true = y_true[rows_to_keep]
    
    X = X.fillna(X.median())
    X = X.values
    
    # Filtrar valores infinitos
    finite_mask = np.isfinite(X).all(axis=1)
    X = X[finite_mask]
    if y_true is not None:
        y_true = y_true[finite_mask]
    
    print(f"Datos después de limpieza: {X.shape[0]} filas, {X.shape[1]} características")
    
    # Escalar los datos
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled[..., np.newaxis]  # Añadir dimensión para CNN 1D
    
    # Hacer predicciones
    print("Haciendo predicciones...")
    predictions = model.predict(X_scaled)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Mostrar resultados
    print("\n=== RESULTADOS DE PREDICCIÓN ===")
    
    # Contar predicciones por clase
    unique_predictions, counts = np.unique(predicted_classes, return_counts=True)
    
    print("\nDistribución de predicciones:")
    for pred_class, count in zip(unique_predictions, counts):
        class_name = class_names[pred_class] if pred_class < len(class_names) else f"Clase_{pred_class}"
        percentage = (count / len(predicted_classes)) * 100
        print(f"  {class_name}: {count} muestras ({percentage:.2f}%)")
    
    # Si tenemos etiquetas verdaderas, calcular precisión
    if y_true is not None:
        try:
            # Cargar label encoder si existe
            le = joblib.load('label_encoder.save')
            y_true_encoded = le.transform(y_true)
            
            # Calcular precisión
            accuracy = np.mean(predicted_classes == y_true_encoded)
            print(f"\nPrecisión del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Mostrar algunas predicciones vs realidad
            print("\nEjemplos de predicciones:")
            print("Real -> Predicho")
            for i in range(min(10, len(y_true))):
                real_name = y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i]
                pred_name = class_names[predicted_classes[i]]
                confidence = np.max(predictions[i]) * 100
                print(f"{real_name} -> {pred_name} (confianza: {confidence:.1f}%)")
                
        except Exception as e:
            print(f"No se pudo calcular la precisión: {e}")
    
    return predictions, predicted_classes, class_names

if __name__ == "__main__":
    # Ejecutar la predicción
    predecir_con_csv()