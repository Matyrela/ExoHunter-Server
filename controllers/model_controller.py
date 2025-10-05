from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
from models import ONED_CNN
from training import predict_exoplanet
from process_data.filter import rename_df, ensure_hostname
import tempfile

router = APIRouter()

@router.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    #if file.content_type != "text/csv":
    #    raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    #if not file.filename.endswith(".csv"):
    #    raise HTTPException(status_code=400, detail="El archivo no tiene extensión .csv")

    try:
        # Guardar el archivo temporalmente
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Leer CSV
        df = pd.read_csv(tmp_path, comment="#")

        # Normalizar nombres de columnas
        df = rename_df(df)

        # Buscar columna de identificación válida
        id_col = None
        for col in ["hostname", "kepler_name", "tid"]:
            if col in df.columns:
                id_col = col
                break

        if not id_col:
            raise HTTPException(
                status_code=400,
                detail="El CSV debe contener al menos una de las siguientes columnas: 'hostname', 'kepler_name' o 'tid'."
            )

        ids = df[id_col].astype(str).tolist()

        # Obtener predicciones
        df = ensure_hostname(df)
        gradient_boost = predict_exoplanet.predict(df)
        cnn = ONED_CNN.probar_modelo(csv_path=tmp_path)

        # Combinar resultados
        result = []
        for idx, (cnn_class, gb_tuple) in enumerate(zip(cnn, gradient_boost)):
            pred, prob = gb_tuple
            result.append({
                "id": ids[idx],
                "class": cnn_class,
                "prediction": int(pred),
                "probability": float(prob),
                "column_id": id_col
            })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el CSV: {e}")

    return result
