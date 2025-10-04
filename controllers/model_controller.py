from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
from models import ONED_CNN

router = APIRouter()

@router.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    #if file.content_type != "text/csv":
    #    raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    #if not file.filename.endswith(".csv"):
    #    raise HTTPException(status_code=400, detail="El archivo no tiene extensión .csv")

    #contents = await file.read()
    try:
        #s = str(contents, 'utf-8')
        #data = StringIO(s)
        #df = pd.read_csv(data, sep=',')
        #df = pd.read_csv('training_csv.csv')
        cnn = ONED_CNN.probar_modelo(csv_path='training_csv.csv')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el CSV: {e}")
    return {
        "1D-CNN": cnn
    }
