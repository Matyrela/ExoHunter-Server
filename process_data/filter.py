import pandas as pd
import argparse

def label_from_disposition(value: str):
    """Etiqueta: Confirmed/Published Confirmed => 1; False Positive/Refuted/Not Dispositioned => 0; Candidate => None (descarta)."""
    if pd.isna(value):
        return None
    u = str(value).upper().strip()
    if "CANDIDATE" in u:
        return None
    if "CONFIRMED" in u or "PUBLISHED CONFIRMED" in u:
        return 1
    if "FALSE POSITIVE" in u or "REFUTED" in u or "NOT DISPOSITIONED" in u:
        return 0
    return None

FEATURES = [
    "sy_snum","sy_pnum","pl_orbper","pl_orbsmax",
    "pl_orbeccen","pl_rade","pl_bmasse","pl_insol","pl_eqt","ttv_flag",
    "st_teff","st_rad","st_mass","st_met","st_logg",
    "sy_dist","sy_vmag","sy_kmag","sy_gaiamag",
    "hostname","label"
]

def find_col(case_insensitive_cols, candidates):
    """Devuelve el nombre real de la primera columna candidata encontrada (case-insensitive)."""
    lowmap = {c.lower().strip(): c for c in case_insensitive_cols}
    for cand in candidates:
        k = cand.lower().strip()
        if k in lowmap:
            return lowmap[k]
    return None

def ensure_hostname(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que exista 'hostname'. Si no existe, lo crea desde KOI name.
       Si existe pero tiene NaN, rellena con KOI name cuando haya."""
    host_col = find_col(df.columns, ["hostname"])
    koi_col  = find_col(df.columns, ["koi_name", "kepoi_name", "koi name", "kepoi name"])
    if host_col is None and koi_col is not None:
        df["hostname"] = df[koi_col]
        print(f"[HOSTNAME] No había 'hostname'. Se creó desde '{koi_col}'.")
    elif host_col is not None and koi_col is not None:
        n_before = df[host_col].isna().sum()
        df[host_col] = df[host_col].fillna(df[koi_col])
        n_after = df[host_col].isna().sum()
        if n_before != n_after:
            print(f"[HOSTNAME] Relleno NaN en '{host_col}' con '{koi_col}': {n_before - n_after} filas completadas.")
    else:
        if host_col is None:
            print("[HOSTNAME] No se encontró 'hostname' ni KOI name para crearlo.")
        # si host_col existe y no hay koi_col, no hacemos nada
    return df

def process_one_csv(path, dedupe_by=None):
    print(f"\n[CARGA] Leyendo: {path}")
    df = pd.read_csv(path, comment="#", low_memory=False)
    print(f"[INFO] Filas iniciales: {len(df)} | Columnas: {len(df.columns)}")

    # Filtrado default_flag por archivo (si existe)
    default_col = find_col(df.columns, ["default_flag"])
    if default_col:
        before = len(df)
        df = df[df[default_col] == 1]
        print(f"[FILTRO] {default_col} == 1 → {len(df)} filas (descartadas: {before - len(df)})")
    else:
        print("[FILTRO] No existe 'default_flag' en este archivo → se conserva todo.")

    # Normalizar/crear hostname desde KOI name si hace falta
    df = ensure_hostname(df)

    # Detectar columna de disposición/soltype
    disp_col = find_col(df.columns, ["disposition", "archive_disposition", "archive disposition", "soltype", "koi_disposition"])
    if not disp_col:
        raise SystemExit("❌ No se encontró ninguna columna de disposición ['disposition', 'archive_disposition', 'soltype'].")

    # Crear label y filtrar candidatos / NaN
    df["label"] = df[disp_col].apply(label_from_disposition)
    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    print(f"[LABEL] Sin candidatos/NaN: {len(df)} (descartadas: {before - len(df)}) | counts: {df['label'].value_counts().to_dict()}")

    # Selección de features presentes
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"[WARN] Faltan columnas en este archivo y se omiten: {missing}")
    cols_present = [c for c in FEATURES if c in df.columns]
    out = df[cols_present].copy()
    print(f"[SELECCIÓN] Columnas usadas: {len(cols_present)} | Filas: {len(out)}")

    # Dedupe por clave opcional (local)
    if dedupe_by:
        keys = [c.strip() for c in dedupe_by.split(",") if c.strip() and c in out.columns]
        if keys:
            b = len(out)
            out = out.drop_duplicates(subset=keys, keep="first")
            print(f"[DEDUPE-KEY local] {keys} → {len(out)} (descartadas: {b - len(out)})")
        else:
            print("[DEDUPE-KEY local] Sin claves válidas presentes en este archivo.")

    # Dedupe exacto local
    b = len(out)
    out = out.drop_duplicates(keep="first")
    if len(out) != b:
        print(f"[DEDUPE-EXACT local] → {len(out)} (descartadas: {b - len(out)})")

    return out

def main(input_csvs, output_csv, dedupe_by=None):
    parts = [process_one_csv(p, dedupe_by=dedupe_by) for p in input_csvs]
    if not parts:
        raise SystemExit("No se pudo procesar ningún archivo.")
    df = pd.concat(parts, ignore_index=True)
    print(f"\n[COMBINE] Total concatenado: {len(df)} filas | Columnas: {len(df.columns)}")

    # Dedupe global por clave (opcional)
    if dedupe_by:
        keys = [c.strip() for c in dedupe_by.split(",") if c.strip() and c in df.columns]
        if keys:
            b = len(df)
            df = df.drop_duplicates(subset=keys, keep="first")
            print(f"[DEDUPE-KEY global] {keys} → {len(df)} (descartadas: {b - len(df)})")
        else:
            print("[DEDUPE-KEY global] Sin claves válidas presentes en el combinado.")

    # Dedupe exacto global
    b = len(df)
    df = df.drop_duplicates(keep="first")
    if len(df) != b:
        print(f"[DEDUPE-EXACT global] → {len(df)} (descartadas: {b - len(df)})")

    # Resumen final
    if "label" in df.columns:
        print("[RESUMEN] Label counts:", df["label"].value_counts(dropna=False).to_dict())

    df.to_csv(output_csv, index=False)
    print(f"[OK] Guardado en: {output_csv} | Filas finales: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_csvs", nargs="+", required=True,
                        help="Uno o más CSV (p. ej., cumulative.csv k2pandc.csv toi.csv)")
    parser.add_argument("--out", dest="output_csv", required=True, help="CSV de salida filtrado")
    parser.add_argument("--dedupe-by", dest="dedupe_by", default=None,
                        help="Claves para deduplicar (coma). Ej: 'hostname' o 'hostname,pl_orbper'")
    args = parser.parse_args()
    main(args.input_csvs, args.output_csv, args.dedupe_by)
