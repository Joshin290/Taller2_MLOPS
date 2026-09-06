# Taller Desarrollo en Contenedores

Requisitos

Para ejecutar el proyecto solo es necesario tener instalados:

- Docker

- Docker Compose

No es necesario instalar Python, JupyterLab, FastAPI ni uv directamente en el sistema operativo anfitrión, ya que estas herramientas se instalan dentro de las imágenes Docker.


Dependencias del proyecto

Las principales dependencias definidas en pyproject.toml son:

fastapi

uvicorn

jupyterlab

scikit-learn

joblib

numpy

pandas

uv.lock mantiene bloqueadas las versiones utilizadas para obtener un entorno reproducible.



Construcción y ejecución

Desde la carpeta proyecto/ ejecutar:

docker compose up --build

Docker Compose construirá las dos imágenes y levantará los servicios.

También es posible ejecutarlos en segundo plano

Para revisar los contenedores activos:

docker compose ps -a


Para detener y eliminar los contenedores:

docker compose down


Servicio JupyterLab

JupyterLab queda disponible en:

http://localhost:8025

El servicio se ejecuta con:

uv run jupyter lab \
  --ip=0.0.0.0 \
  --port=8025 \
  --no-browser \
  --allow-root \
  --NotebookApp.token=''

El notebook principal es:

Entrenamiento.ipynb

Entrenamiento de modelos


El notebook utiliza el dataset Iris incluido en scikit-learn:

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)


Se entrenan tres algoritmos.

1. Random Forest

RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

Archivo generado:

modelos/random_forest.pkl

2. Support Vector Classifier

SVC(
    kernel="rbf",
    random_state=42
)

Archivo generado:

modelos/svc.pkl

3. K-Nearest Neighbors

KNeighborsClassifier(n_neighbors=5)



Archivo generado:

modelos/knn.pkl

Además, inicialmente Random Forest es guardado como modelo activo:

modelos/modelo_activo.pkl
