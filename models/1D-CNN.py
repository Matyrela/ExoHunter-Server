import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Cargar datos saltando las líneas de comentario (empiezan con #)
df = pd.read_csv('PS_2025.10.02_11.56.01.csv', comment='#')
print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns[:10])}...")  # Mostrar primeras 10 columnas

# Seleccionar solo columnas numéricas para características
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nColumnas numéricas encontradas: {len(numeric_columns)}")
print(f"Primeras columnas numéricas: {numeric_columns[:10]}")

# Usar 'discoverymethod' como variable objetivo (categórica)
# Primero, eliminarla de las características si está presente
if 'discoverymethod' in df.columns:
    y = df['discoverymethod']
    # Remover discoverymethod de las columnas numéricas si está ahí
    if 'discoverymethod' in numeric_columns:
        numeric_columns.remove('discoverymethod')
else:
    # Si no está discoverymethod, usar la primera columna categórica disponible
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        target_col = categorical_cols[0]
        y = df[target_col]
        print(f"Usando '{target_col}' como variable objetivo")
    else:
        # Usar última columna como objetivo
        y = df.iloc[:, -1]
        print("Usando última columna como variable objetivo")

# Usar solo las columnas numéricas como características
X = df[numeric_columns]
print(f"\nCaracterísticas seleccionadas: {X.shape[1]} columnas")
print(f"Variable objetivo: {len(y.unique())} clases únicas")

# Manejar valores NaN de manera más robusta
print(f"\nValores NaN antes de limpiar: {X.isnull().sum().sum()}")

# Remover columnas con más del 70% de valores faltantes
threshold = 0.7
columns_to_keep = []
for col in X.columns:
    missing_ratio = X[col].isnull().sum() / len(X)
    if missing_ratio < threshold:
        columns_to_keep.append(col)

X = X[columns_to_keep]
print(f"Columnas mantenidas después de filtrar por valores faltantes: {len(columns_to_keep)}")

# Remover filas con más del 50% de valores faltantes
row_threshold = len(X.columns) * 0.5
rows_to_keep = X.isnull().sum(axis=1) < row_threshold
X = X[rows_to_keep]
y = y[rows_to_keep]
print(f"Filas mantenidas: {X.shape[0]}")

# Rellenar valores NaN restantes con la mediana (más robusta que la media)
X = X.fillna(X.median())
print(f"Valores NaN después de limpiar: {X.isnull().sum().sum()}")

# Convertir a numpy array
X = X.values

# Verificar y limpiar valores infinitos
finite_mask = np.isfinite(X).all(axis=1)
X = X[finite_mask]
y = y[finite_mask]
print(f"Filas después de remover infinitos: {X.shape[0]}")

# Codificar etiquetas si son categóricas
if y.dtype == 'O' or len(np.unique(y)) < 20:
	le = LabelEncoder()
	y = le.fit_transform(y)
	class_names = le.classes_
else:
	class_names = np.unique(y)
y_cat = to_categorical(y)

# Normalizar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Ajustar dimensiones para 1D-CNN
X = X[..., np.newaxis]

# Separar en 70% train, 15% test, 15% validación
X_temp, X_test, y_temp, y_test = train_test_split(X, y_cat, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)
# 0.1765 ≈ 15/(85), para que validación sea 15% del total

# Definir modelo 1D-CNN
model = Sequential([
	Conv1D(64, 3, activation='relu', input_shape=(X.shape[1], 1)),
	MaxPooling1D(2),
	Dropout(0.3),
	Conv1D(128, 3, activation='relu'),
	MaxPooling1D(2),
	Flatten(),
	Dense(64, activation='relu'),
	Dropout(0.3),
	Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar modelo
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Evaluar modelo
loss, acc = model.evaluate(X_test, y_test)
print(f'Precisión en test: {acc:.2f}')

# Predecir
y_pred = model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Matriz de confusión y reporte
print("Matriz de confusión:")
print(confusion_matrix(y_test_labels, y_pred_labels))
print("\nReporte de clasificación:")

# Usar solo las clases que aparecen en el conjunto de test
unique_test_labels = np.unique(y_test_labels)
class_names_present = [str(class_names[i]) for i in unique_test_labels]
print(classification_report(y_test_labels, y_pred_labels, labels=unique_test_labels, target_names=class_names_present))
print(f"Precisión final: {accuracy_score(y_test_labels, y_pred_labels):.4f}")
