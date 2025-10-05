from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controllers import model_controller

app = FastAPI()

origins = [
    "http://localhost:4200",
    "https://exohunter.earth"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_controller.router, prefix="/api", tags=["model"])

@app.get("/")
def root():
    return {"message": "La API está activa"}

