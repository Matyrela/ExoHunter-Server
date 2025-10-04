import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# Workaround para joblib + FastAPI multiproceso
from training.train_exoplanet_lgbm import Log1pTransformer
import sys
sys.modules['__main__'].Log1pTransformer = Log1pTransformer


# ---------- Script de predicción ----------

def predict(input_csv: Path):
    # 1) Cargar modelo (bundle con preprocessor + model)
    model_path = 'training/exoplanet_lgbm.pkl'
    bundle = joblib.load(model_path)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]

    # 2) Cargar CSV nuevo
    df = pd.read_csv(input_csv, comment="#", low_memory=False)

    # 3) Seleccionar columnas que el modelo conoce
    features = preprocessor.feature_names_in_.tolist()
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"El CSV de entrada no tiene estas columnas necesarias: {missing}")

    X = df[features].copy()

    # 4) Transformar con preprocessor
    X_t = preprocessor.transform(X)

    # 5) Predicciones
    probs = model.predict_proba(X_t)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # 6) Devolver resultados en lista
    result = []
    for i in range(len(probs)):
        result.append((int(preds[i]), float(probs[i])))
    return result

