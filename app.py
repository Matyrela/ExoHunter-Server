from fastapi import FastAPI
from controllers import model_controller

app = FastAPI()

app.include_router(model_controller.router, prefix="/api", tags=["model"])

@app.get("/")
def root():
    return {"message": "La API está activa"}

