import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import os


def preparar_datos(csv_path='PS_2025.10.02_11.56.01.csv'):
    df = pd.read_csv(csv_path, comment='#')
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'discoverymethod' in df.columns:
        y = df['discoverymethod']
        if 'discoverymethod' in numeric_columns:
            numeric_columns.remove('discoverymethod')
    else:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            target_col = categorical_cols[0]
            y = df[target_col]
        else:
            y = df.iloc[:, -1]
    X = df[numeric_columns]
    threshold = 0.7
    columns_to_keep = [col for col in X.columns if X[col].isnull().sum() / len(X) < threshold]
    X = X[columns_to_keep]
    row_threshold = len(X.columns) * 0.5
    rows_to_keep = X.isnull().sum(axis=1) < row_threshold
    X = X[rows_to_keep]
    y = y[rows_to_keep]
    X = X.fillna(X.median())
    X = X.values
    finite_mask = np.isfinite(X).all(axis=1)
    X = X[finite_mask]
    y = y[finite_mask]
    if y.dtype == 'O' or len(np.unique(y)) < 20:
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_names = le.classes_
    else:
        class_names = np.unique(y)
    y_cat = to_categorical(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X[..., np.newaxis]
    # Guardar scaler y label encoder
    joblib.dump(scaler, 'scaler.save')
    if 'le' in locals():
        joblib.dump(le, 'label_encoder.save')
    np.save('class_names.npy', class_names)
    return X, y_cat, class_names

def entrenar_y_guardar_modelo(csv_path='PS_2025.10.02_11.56.01.csv', model_path='modelo_1dcnn.h5'):
    X, y_cat, class_names = preparar_datos(csv_path)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_cat, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)
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
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))
    model.save(model_path)
    np.savez('test_data.npz', X_test=X_test, y_test=y_test)
    print('Modelo y datos guardados.')
    return model, history

def probar_modelo(model_path='modelo_1dcnn.h5', csv_path='test_data.npz'):
    model = load_model(model_path)
    class_names = np.load('class_names.npy', allow_pickle=True)
    scaler = joblib.load('scaler.save')
    if csv_path.endswith('.csv'):
        df = pd.read_csv(csv_path, comment='#')
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'discoverymethod' in df.columns:
            if 'discoverymethod' in numeric_columns:
                numeric_columns.remove('discoverymethod')
        X = df[numeric_columns]
        threshold = 0.7
        columns_to_keep = [col for col in X.columns if X[col].isnull().sum() / len(X) < threshold]
        X = X[columns_to_keep]
        row_threshold = len(X.columns) * 0.5
        rows_to_keep = X.isnull().sum(axis=1) < row_threshold
        X = X[rows_to_keep]
        X = X.fillna(X.median())
        X = X.values
        finite_mask = np.isfinite(X).all(axis=1)
        X = X[finite_mask]
        X_scaled = scaler.transform(X)
        X_scaled = X_scaled[..., np.newaxis]
        predictions = model.predict(X_scaled)
        predicted_classes = np.argmax(predictions, axis=1)
        return [class_names[i] for i in predicted_classes]
    else:
        data = np.load(csv_path)
        X_test = data['X_test']
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        return [class_names[i] for i in predicted_classes]
