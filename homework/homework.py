# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
import zipfile


def clean_dataset(path):
    # Ruta al archivo ZIP
    zip_path = path

    with zipfile.ZipFile(zip_path, "r") as z:
        csv_file = z.namelist()[0]
        with z.open(csv_file) as f:
            df = pd.read_csv(f)

    # 1. Renombrar la columna "default payment next month" a "default"
    df.rename(columns={"default payment next month": "default"}, inplace=True)

    # 2. Remover la columna "ID"
    df.drop(columns=["ID"], inplace=True)

    # 3. Eliminar registros con información no disponible (NaN)
    df.dropna(inplace=True)

    # 4. Agrupar valores de EDUCATION > 4 como "others"
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 4 else 4)

    return df


df_test = clean_dataset("./files/input/test_data.csv.zip")
df_train = clean_dataset("./files/input/train_data.csv.zip")

x_train = df_train.drop(columns=["default"])
y_train = df_train["default"]

x_test = df_test.drop(columns=["default"])
y_test = df_test["default"]


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def build_pipeline():
    """
    Construye un pipeline de preprocesamiento y clasificación con Random Forest.

    Returns:
        Pipeline: pipeline entrenable.
    """
    # Variables categóricas a transformar
    categorical_features = ["EDUCATION", "MARRIAGE", "SEX"]

    # One-Hot Encoding para variables categóricas
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Transformador para preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[("cat", categorical_transformer, categorical_features)],
        remainder="passthrough",  # Mantener otras columnas sin cambios
    )

    # Modelo Random Forest
    random_forest_model = RandomForestClassifier(random_state=42)

    # Construcción del pipeline
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", random_forest_model)]
    )

    return pipeline


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score


def optimize_pipeline(pipeline, x_train, y_train):
    """
    Optimiza el pipeline usando GridSearchCV y validación cruzada con 10 folds.

    Args:
        pipeline (Pipeline): Pipeline de clasificación.
        x_train (DataFrame): Datos de entrenamiento (variables independientes).
        y_train (Series): Etiquetas de entrenamiento.

    Returns:
        GridSearchCV: Objeto GridSearchCV optimizado.
    """
    # Definir métrica de optimización
    scoring = make_scorer(balanced_accuracy_score)

    # Espacio de hiperparámetros
    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
    }

    # GridSearch con validación cruzada (10 folds)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=10, scoring=scoring, n_jobs=-1, verbose=2
    )

    # Ajustar el modelo
    grid_search.fit(x_train, y_train)

    return grid_search


import pickle
import gzip
import os


def save_model(model, file_path="files/models/model.pkl.gz"):
    """
    Guarda el modelo en un archivo comprimido con gzip.

    Args:
        model (Pipeline): Modelo entrenado.
        file_path (str): Ruta donde se guardará el modelo.
    """
    os.makedirs(
        os.path.dirname(file_path), exist_ok=True
    )  # Crear directorio si no existe

    with gzip.open(file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modelo guardado en {file_path}")


from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
)
import json


import json
import os


import os
import json
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


import os
import json
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(
    model, x_train, y_train, x_test, y_test, file_path="files/output/metrics.json"
):
    """
    Evalúa el modelo en los conjuntos de entrenamiento y prueba y guarda las métricas y matrices de confusión en un archivo JSON.

    Args:
        model (Pipeline): Modelo entrenado.
        x_train, y_train: Datos de entrenamiento.
        x_test, y_test: Datos de prueba.
        file_path (str): Ruta del archivo JSON donde se guardarán las métricas.
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Abrir archivo en modo escritura
    with open(file_path, "w") as f:
        # Predicciones
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Métricas de entrenamiento
        train_metrics = {
            "type": "metrics",
            "dataset": "train",
            "precision": precision_score(y_train, y_train_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
            "recall": recall_score(y_train, y_train_pred, zero_division=0),
            "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
        }
        f.write(json.dumps(train_metrics) + "\n")

        # Métricas de prueba
        test_metrics = {
            "type": "metrics",
            "dataset": "test",
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
        }
        f.write(json.dumps(test_metrics) + "\n")

        # Matrices de confusión
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)

        train_cm = {
            "type": "cm_matrix",
            "dataset": "train",
            "true_0": {
                "predicted_0": int(cm_train[0][0]),
                "predicted_1": int(cm_train[0][1]),
            },
            "true_1": {
                "predicted_0": int(cm_train[1][0]),
                "predicted_1": int(cm_train[1][1]),
            },
        }
        f.write(json.dumps(train_cm) + "\n")

        test_cm = {
            "type": "cm_matrix",
            "dataset": "test",
            "true_0": {
                "predicted_0": int(cm_test[0][0]),
                "predicted_1": int(cm_test[0][1]),
            },
            "true_1": {
                "predicted_0": int(cm_test[1][0]),
                "predicted_1": int(cm_test[1][1]),
            },
        }
        f.write(json.dumps(test_cm) + "\n")

    print(f"Métricas y matrices de confusión guardadas correctamente en {file_path}")


# Construcción del Pipeline
pipeline = build_pipeline()

# Optimización del modelo
best_pipeline = optimize_pipeline(pipeline, x_train, y_train)

# Guardado del modelo
save_model(best_pipeline)

# Evaluación del modelo y guardado de métricas
evaluate_model(best_pipeline, x_train, y_train, x_test, y_test)
