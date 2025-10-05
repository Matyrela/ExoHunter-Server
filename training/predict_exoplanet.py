import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

from training.train_exoplanet_lgbm import Log1pTransformer
import sys
sys.modules['__main__'].Log1pTransformer = Log1pTransformer


# ---------- Script de predicción ----------

def predict(df: pd.DataFrame):
    # 1) Cargar modelo (bundle con preprocessor + model)
    model_path = 'training/exoplanet_lgbm.pkl'
    bundle = joblib.load(model_path)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]

    # 2) Seleccionar columnas que el modelo conoce
    features = preprocessor.feature_names_in_.tolist()
    # Agregar columnas faltantes como np.nan
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
    X = df[features].copy()

    # 3) Transformar con preprocessor
    X_t = preprocessor.transform(X)

    # 4) Predicciones
    probs = model.predict_proba(X_t)[:, 1]
    preds = (probs >= 0.4).astype(int)

    # 5) Devolver resultados en lista
    result = []
    for i in range(len(probs)):
        result.append((int(preds[i]), float(probs[i])))
    return result
