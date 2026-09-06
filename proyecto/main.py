from enum import IntEnum
from pathlib import Path
import shutil
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# Configuración de Rutas y Constantes
# ------------------------------------------------------------------------------
MODELS_DIR = Path("./modelos")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# Definición de Menú/Opciones con IntEnum
# ------------------------------------------------------------------------------
class ModelOption(IntEnum):
    RANDOM_FOREST = 1
    SVC = 2
    KNN = 3


MODEL_MAP = {
    ModelOption.RANDOM_FOREST: "random_forest",
    ModelOption.SVC: "svc",
    ModelOption.KNN: "knn",
}


# ------------------------------------------------------------------------------
# Esquema Pydantic con Texto Descriptivo
# ------------------------------------------------------------------------------
class ModelSelection(BaseModel):
    model_id: ModelOption = Field(
        ...,
        title="ID del Modelo",
        description="Número del modelo a activar:\n- 1: Random Forest\n- 2: Support Vector Machine (SVC)\n- 3: K-Nearest Neighbors (KNN)",
        example=1,
    )


class IrisPredictionInput(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


# ------------------------------------------------------------------------------
# Aplicación FastAPI
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Taller 2 MLOps",
    description="API para la inferencia y cambio dinámico de modelos de Machine Learning.",
    version="1.0.0",
)


@app.get("/", tags=["General"])
def root():
    return {"message": "Servicio de Inferencia de Modelos MLOps activo"}


@app.get("/model", tags=["Modelo"])
def get_active_model():
    """Retorna la información del modelo que está activo actualmente."""
    active_file = MODELS_DIR / "modelo_activo.pkl"

    if not active_file.exists():
        raise HTTPException(
            status_code=404,
            detail="No se encontró ningún modelo activo publicado en './modelos/modelo_activo.pkl'.",
        )

    try:
        model = joblib.load(active_file)
        return {
            "active_model_file": active_file.name,
            "algorithm_class": type(model).__name__,
            "details": str(model),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al leer el modelo activo: {str(e)}"
        )


@app.post("/select-model", tags=["Modelo"])
def select_model(selection: ModelSelection):
    """
    Cambia dinámicamente el modelo activo enviando su número de identificación:

    - **1**: Random Forest
    - **2**: Support Vector Classifier (SVC)
    - **3**: K-Nearest Neighbors (KNN)
    """
    selected_enum = selection.model_id

    if selected_enum not in MODEL_MAP:
        raise HTTPException(
            status_code=400,
            detail="Opción inválida. Opciones disponibles: 1 (Random Forest), 2 (SVC), 3 (KNN).",
        )

    selected_name = MODEL_MAP[selected_enum]
    source_file = MODELS_DIR / f"{selected_name}.pkl"
    active_file = MODELS_DIR / "modelo_activo.pkl"

    if not source_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"El archivo '{source_file.name}' no existe en '{MODELS_DIR}'.",
        )

    try:
        shutil.copy(source_file, active_file)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al copiar el modelo seleccionado: {str(e)}",
        )

    return {
        "message": "Modelo actualizado exitosamente",
        "selected_id": int(selected_enum),
        "active_model": selected_name,
    }


@app.post("/predict", tags=["Inferencia"])
def predict(features: IrisPredictionInput):
    """Realiza una predicción utilizando el modelo activo actual."""
    active_file = MODELS_DIR / "modelo_activo.pkl"

    if not active_file.exists():
        raise HTTPException(
            status_code=404, detail="No hay ningún modelo activo cargado."
        )

    try:
        model = joblib.load(active_file)
        input_data = [
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ]

        prediction = model.predict(input_data)[0]
        class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}

        return {
            "prediction_code": int(prediction),
            "predicted_class": class_names.get(int(prediction), str(prediction)),
            "model_used": type(model).__name__,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error durante la inferencia: {str(e)}"
        )