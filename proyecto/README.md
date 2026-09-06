# Taller 2 - MLOps

## Grupo 3

### Integrantes
* **Juan David Clavijo Ortiz**
* **Laura Sofía Rodríguez Pérez**
* **Joshua Alexander Valero Lozano**

---

Este repositorio contiene como objetivo construir un ambiente de desarrollo para Machine Learning utilizando Docker Compose, uv, JupyterLab y FastAPI.

La solución está compuesta por dos servicios ejecutados mediante Docker Compose:

JupyterLab, utilizado para entrenar y guardar modelos de Machine Learning.

API FastAPI, utilizada para consumir los modelos entrenados y realizar inferencia.

Ambos servicios comparten el mismo directorio del proyecto mediante un volumen, permitiendo que un modelo generado o actualizado desde JupyterLab pueda ser utilizado inmediatamente por la API.

Estructura general

Taller2_MLOPS/
├── Indicaciones.md
├── README.md
└── proyecto/
    ├── Dockerfile.api
    ├── Dockerfile.jupyter
    ├── docker-compose.yaml
    ├── Entrenamiento.ipynb
    ├── main.py
    ├── pyproject.toml
    ├── uv.lock
    ├── README.md
    ├── modelos/
    │   ├── random_forest.pkl
    │   ├── svc.pkl
    │   ├── knn.pkl
    │   └── modelo_activo.pkl
    └── src/

Ejecución rápida

Ingresar al directorio del proyecto:

cd proyecto

Construir e iniciar los servicios:

docker compose up --build

Una vez iniciados los contenedores:

JupyterLab: http://localhost:8025

API FastAPI: http://localhost:8000

••OpenAPI: http://localhost:8000/docs

Para detener los servicios:

docker compose down

Funcionamiento

El notebook Entrenamiento.ipynb utiliza el dataset Iris de scikit-learn para entrenar tres clasificadores:

Random Forest

Support Vector Classifier (SVC)

K-Nearest Neighbors (KNN)

Los modelos son almacenados en la carpeta compartida modelos/.

La API permite consultar el modelo activo, seleccionar dinámicamente uno de los modelos disponibles y realizar predicciones utilizando el modelo seleccionado.