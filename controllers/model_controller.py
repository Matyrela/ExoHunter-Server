from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
from models import ONED_CNN
from training import predict_exoplanet
import tempfile

router = APIRouter()

@router.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    #if file.content_type != "text/csv":
    #    raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    #if not file.filename.endswith(".csv"):
    #    raise HTTPException(status_code=400, detail="El archivo no tiene extensión .csv")

    try:
        # Guardar el archivo subido en un archivo temporal
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        # Leer el CSV para obtener los ids reales
        df = pd.read_csv(tmp_path, comment='#')
        if 'id_name' not in df.columns:
            raise HTTPException(status_code=400, detail="El CSV no contiene la columna 'id_name'")
        ids = df['id_name'].tolist()
        # Obtener predicciones de ambos modelos
        gradientBoost = predict_exoplanet.predict(input_csv=tmp_path)
        cnn = ONED_CNN.probar_modelo(csv_path=tmp_path)
        # Unificar resultados usando el id_name real
        result = []
        for idx, (cnn_class, gb_tuple) in enumerate(zip(cnn, gradientBoost)):
            prediccion, probabilidad = gb_tuple
            result.append({
                "id": ids[idx],
                "clase": cnn_class,
                "prediccion": int(prediccion),
                "probabilidad": float(probabilidad)
            })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el CSV: {e}")
    return result
