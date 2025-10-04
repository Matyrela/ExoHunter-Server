from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
from models import ONED_CNN
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
        # Pasar la ruta real del archivo temporal a probar_modelo
        cnn = ONED_CNN.probar_modelo(csv_path=tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el CSV: {e}")
    return {
        "1D-CNN": cnn
    }
