from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib

app = FastAPI(
    title="Taller 2 MLOps",
    description="Uso de UV y Docker-Compose para desplegar modelo de ML en un contenedor con FastAPI y Jupyter"
)

MODEL_PATH = Path("./modelos/modelo_activo.pkl")

class IrisInput(BaseModel):
    sepal_length: float = 5.1
    sepal_width: float = 3.5
    petal_length: float = 1.4
    petal_width: float = 0.2


# Endpoint GET 1: Consultar el modelo activo en disco
@app.get("/model")
def get_active_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="No hay modelo activo cargado.")
    
    model = joblib.load(MODEL_PATH)
    return {
        "modelo_activo": model.__class__.__name__
    }
# Endpoint POST: Inferencia
@app.post("/predict")
def predict(data: IrisInput):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No hay ningún modelo activo en './modelos/modelo_activo.pkl'."
        )
    
    # Lectura dinámica del modelo sobreescrito
    model = joblib.load(MODEL_PATH)
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = int(model.predict(features)[0])
    
    return {
        "status": "success",
        "modelo_elegido": model.__class__.__name__,
        "prediccion": prediction
    }