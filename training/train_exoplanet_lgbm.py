r"""
Entrenamiento de un clasificador LightGBM para exoplanetas.
COMO USAR (PowerShell):
python .\train_exoplanet_lgbm.py `
  --csv .\exoplanet_training.csv `
  --label-col label `
  --group-col hostname `
  --out-model exoplanet_lgbm.pkl `
  --out-report training_report.txt `
  --plots
"""

import argparse
from pathlib import Path
from typing import List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# ---------- Transformador custom (picklable) ----------

LOG1P_CANDIDATES = ["pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "pl_insol", "sy_dist"]

class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log1p_idx=None):
        self.log1p_idx = log1p_idx or []

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        return self

    def set_output(self, *, transform=None):
        # No cambiamos el tipo de salida; sólo guardamos la preferencia por si la querés usar
        self._transform_output = transform
        return self

    def transform(self, X):
        # Asegurar ndarray aunque llegue un DataFrame por set_output("pandas")
        if hasattr(X, "to_numpy"):
            X_arr = X.to_numpy(dtype="float64", copy=True)
        else:
            X_arr = np.asarray(X, dtype="float64")

        for j in self.log1p_idx:
            col = X_arr[:, j]
            neg_mask = col < 0
            if np.any(neg_mask):
                col[neg_mask] = np.nan
            X_arr[:, j] = np.log1p(col)

        return X_arr


# ---------- Funciones auxiliares ----------

def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    log1p_idx = [num_cols.index(c) for c in LOG1P_CANDIDATES if c in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", Log1pTransformer(log1p_idx)),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    # Intentar salida pandas para evitar warnings de nombres en LightGBM
    try:
        ct.set_output(transform="pandas")
    except Exception as e:
        print(f"[Warn] No se pudo set_output('pandas') en ColumnTransformer: {e}. Uso de salida NumPy.")

    return ct


def pick_columns(df: pd.DataFrame, label_col: str, group_col: Optional[str] = None):
    drop_cols = {label_col}
    if group_col:
        drop_cols.add(group_col)

    num_cols, cat_cols = [], []
    for c in df.columns:
        if c in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def split_data_three(X, y, group_series=None, seed=42):
    """Divide en train (70%), val (15%), test (15%), respetando grupos si se proveen."""
    if group_series is not None:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_idx, temp_idx = next(gss1.split(X, y, groups=group_series))
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
        group_temp = group_series.iloc[temp_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=group_temp))
        return (
            X_train, X_temp.iloc[val_idx], X_temp.iloc[test_idx],
            y_train, y_temp.iloc[val_idx], y_temp.iloc[test_idx]
        )
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
        )
        return X_train, X_val, X_test, y_train, y_val, y_test


def best_threshold(y_true, proba, beta=1.0):
    """Busca umbral que maximiza F1-macro en valid."""
    ths = np.linspace(0.05, 0.95, 19)
    scores = []
    for t in ths:
        y_pred = (proba >= t).astype(int)
        scores.append((f1_score(y_true, y_pred, average="macro"), t))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[0]  # (F1, thr)


# ---------- Función principal de entrenamiento ----------

def train(
    csv_path: Path,
    label_col: str = "label",
    group_col: Optional[str] = None,
    out_model: Path = Path("exoplanet_lgbm.pkl"),
    out_report: Path = Path("training_report.txt"),
    save_plots: bool = False,
    seed: int = 42
):
    # 1) Cargar datos
    df = pd.read_csv(csv_path, comment="#", low_memory=False)
    if label_col not in df.columns:
        raise SystemExit(f"El CSV no tiene columna '{label_col}'.")

    y = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
    if y.isna().all():
        raise SystemExit("La columna de etiquetas está vacía o mal definida.")

    # groups robusto: si hay NaN en group_col, asignar un grupo único por fila
    if group_col and group_col in df.columns:
        raw_groups = df[group_col]
        if raw_groups.isna().any():
            filler = "__row_" + df.index.astype(str)
            group_series = raw_groups.astype(object).copy()
            group_series.loc[raw_groups.isna()] = filler.loc[raw_groups.isna()]
            print(f"[Groups] {raw_groups.isna().sum()} filas sin '{group_col}' recibieron grupos únicos por fila.")
        else:
            group_series = raw_groups
    else:
        group_series = None

    # Columnas
    num_cols, cat_cols = pick_columns(df, label_col=label_col, group_col=group_col)
    X = df[num_cols + cat_cols].copy()

    if "ttv_flag" in X.columns:
        X["ttv_flag"] = pd.to_numeric(X["ttv_flag"], errors="coerce")

    # 2) Split 70/15/15
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data_three(
        X, y, group_series=group_series, seed=seed
    )
    print(f"[Split] Train: {len(X_train)} | Valid: {len(X_valid)} | Test: {len(X_test)}")

    # 3) Preprocesador
    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_valid_t = preprocessor.transform(X_valid)

    # 4) Ponderación de clases (en base a y_train)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    # Queremos subir el peso de la clase 0 si es minoritaria
    w_neg = (n_pos / max(n_neg, 1)) if n_neg < n_pos else 1.0
    w_pos = 1.0

    print(f"[ClassWeight] y_train -> pos={n_pos}, neg={n_neg} | weights {{0:{w_neg:.3f}, 1:{w_pos:.3f}}}")

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
        class_weight={0: w_neg, 1: w_pos},
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=200),
        lgb.log_evaluation(100),
    ]

    model.fit(
        X_train_t, y_train,
        eval_set=[(X_valid_t, y_valid)],
        eval_metric="auc",
        callbacks=callbacks,
    )

    # 5) Evaluación en valid
    valid_proba = model.predict_proba(X_valid_t)[:, 1]

    # Métricas a 0.5 (referencia)
    valid_pred_05 = (valid_proba >= 0.5).astype(int)
    roc_val = roc_auc_score(y_valid, valid_proba)
    pr_val  = average_precision_score(y_valid, valid_proba)
    cm_val_05  = confusion_matrix(y_valid, valid_pred_05)
    rep_val_05 = classification_report(y_valid, valid_pred_05, digits=3)

    # Búsqueda de umbral óptimo por F1-macro
    f1_val_best, thr = best_threshold(y_valid, valid_proba)
    valid_pred = (valid_proba >= thr).astype(int)
    cm_val  = confusion_matrix(y_valid, valid_pred)
    rep_val = classification_report(y_valid, valid_pred, digits=3)

    print(f"\n[VALID] ROC-AUC: {roc_val:.4f} | PR-AUC: {pr_val:.4f}")
    print(f"[VALID] Best threshold by F1-macro: {thr:.3f} (F1={f1_val_best:.4f})")
    print("\nMatriz de confusión (VALID) @thr:\n", cm_val)
    print("\nReporte de clasificación (VALID) @thr:\n", rep_val)

    # 6) Guardar modelo
    joblib.dump({"preprocessor": preprocessor, "model": model, "threshold": float(thr)}, out_model)
    print(f"[OK] Modelo guardado en: {out_model}")

    # 7) Importancias
    booster = model.booster_
    importances = booster.feature_importance(importance_type="gain")
    feature_names = list(X_train_t.columns)  # gracias a set_output("pandas")
    safe_names = feature_names[:len(importances)]
    pd.DataFrame({"feature": safe_names, "gain": importances}) \
        .sort_values("gain", ascending=False) \
        .to_csv(Path(out_report).with_suffix(".importances.csv"), index=False)

    # 8) Plots opcionales (de valid)
    if save_plots:
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
        RocCurveDisplay.from_predictions(y_valid, valid_proba)
        plt.savefig(Path(out_report).with_suffix(".roc.png"))
        plt.close()
        PrecisionRecallDisplay.from_predictions(y_valid, valid_proba)
        plt.savefig(Path(out_report).with_suffix(".pr.png"))
        plt.close()

    # 9) Evaluación final en test (usando el umbral óptimo hallado en valid)
    X_test_t = preprocessor.transform(X_test)
    test_proba = model.predict_proba(X_test_t)[:, 1]
    test_pred  = (test_proba >= thr).astype(int)

    roc_test = roc_auc_score(y_test, test_proba)
    pr_test  = average_precision_score(y_test, test_proba)
    cm_test  = confusion_matrix(y_test, test_pred)
    rep_test = classification_report(y_test, test_pred, digits=3)

    print(f"\n[FINAL TEST] ROC-AUC: {roc_test:.4f} | PR-AUC: {pr_test:.4f} | thr={thr:.3f}")
    print("\nMatriz de confusión (TEST) @thr:\n", cm_test)
    print("\nReporte de clasificación (TEST) @thr:\n", rep_test)

    # 10) Guardar reporte completo
    report_txt = (
        f"Archivo: {csv_path}\n"
        f"Filas: {len(df)}\n"
        f"Features num: {len(num_cols)} | cat: {len(cat_cols)}\n"
        "\n=== VALIDATION (15%) ===\n"
        f"ROC-AUC: {roc_val:.6f}\n"
        f"PR-AUC : {pr_val:.6f}\n"
        f"Best threshold (F1-macro): {thr:.3f} | F1: {f1_val_best:.6f}\n"
        f"\n[Matriz @thr]\n{cm_val}\n\n[Reporte @thr]\n{rep_val}\n"
        f"\n[Referencia @0.5]\nMatriz:\n{cm_val_05}\n\nReporte:\n{rep_val_05}\n"
        "\n=== FINAL TEST (15%) ===\n"
        f"ROC-AUC: {roc_test:.6f}\n"
        f"PR-AUC : {pr_test:.6f}\n"
        f"threshold usado: {thr:.3f}\n"
        f"Matriz @thr:\n{cm_test}\n\nReporte @thr:\n{rep_test}\n"
    )
    Path(out_report).write_text(report_txt, encoding="utf-8")
    print(f"[OK] Reporte guardado en: {out_report}")


# ---------- CLI ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path, help="CSV filtrado con features + label")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--out-model", default="exoplanet_lgbm.pkl", type=Path)
    ap.add_argument("--out-report", default="training_report.txt", type=Path)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    train(
        csv_path=args.csv,
        label_col=args.label_col,
        group_col=args.group_col,
        out_model=args.out_model,
        out_report=args.out_report,
        save_plots=args.plots,
    )
