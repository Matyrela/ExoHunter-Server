"""
Predicción con el modelo entrenado LightGBM para exoplanetas.

Uso:
    python predict_exoplanet.py `
        --model exoplanet_lgbm.pkl `
        --csv nuevos_exoplanetas.csv `
        --out predicciones.csv
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from train_exoplanet_lgbm import Log1pTransformer


# ---------- Script de predicción ----------

def predict(model_path: Path, input_csv: Path, output_csv: Path):
    # 1) Cargar modelo (bundle con preprocessor + model)
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



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path, help="Modelo .pkl entrenado")
    ap.add_argument("--csv", required=True, type=Path, help="CSV nuevo con features")
    ap.add_argument("--out", default="predicciones.csv", type=Path, help="Archivo de salida con predicciones")
    args = ap.parse_args()

    print(predict(args.model, args.csv, args.out))
