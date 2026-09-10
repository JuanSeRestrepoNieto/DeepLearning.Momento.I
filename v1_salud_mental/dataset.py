"""
Carga y preprocesamiento del dataset
"AI & Social Media Impact: Student Health & Grades" (Kaggle).

Tarea: clasificacion multiclase del nivel de salud mental del estudiante
(Bajo / Medio / Alto) a partir de sus habitos de vida y uso de tecnologia.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

CSV_PATH = "data/AI_SocialMedia_Student_Dataset.csv"

CLASS_NAMES = ["Bajo", "Medio", "Alto"]

# Columnas que no son predictores
ID_COLUMN     = "Student_ID"
TARGET_COLUMN = "Mental_Health_Score"

CATEGORICAL_COLUMNS = ["Gender", "Education_Level"]

RANDOM_STATE = 42


def load_dataframe(csv_path=CSV_PATH):
    """Lee el CSV y elimina identificadores, nulos y duplicados."""

    df = pd.read_csv(csv_path)

    df = df.drop(columns=[ID_COLUMN])
    df = df.dropna()
    df = df.drop_duplicates()

    return df


def build_target(df):
    """
    Discretiza el puntaje continuo de salud mental en tres clases usando
    terciles: 0 = Bajo, 1 = Medio, 2 = Alto.

    Los cortes se calculan sobre todo el dataset porque esto define el
    problema (la variable Y), no es una transformacion aprendida de las
    features. Los terciles garantizan clases balanceadas.
    """

    y, bins = pd.qcut(df[TARGET_COLUMN], q=3, labels=False, retbins=True)

    X = df.drop(columns=[TARGET_COLUMN])

    return X, y.to_numpy(), bins


def encode_features(X):
    """One-Hot Encoding de las variables categoricas nominales."""

    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, drop_first=False)

    return X.astype(np.float32)


def get_dataloaders(batch_size=64, csv_path=CSV_PATH, verbose=True):
    """
    Devuelve los DataLoaders de entrenamiento, validacion y prueba,
    junto con metadatos utiles para construir y evaluar el modelo.

    Particion estratificada 70 / 15 / 15.
    El StandardScaler se ajusta UNICAMENTE con el conjunto de
    entrenamiento para evitar fuga de informacion (data leakage).
    """

    df = load_dataframe(csv_path)

    X, y, bins = build_target(df)
    X = encode_features(X)

    feature_names = list(X.columns)
    X = X.to_numpy()

    # Primera particion: 70% train, 30% temporal
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Segunda particion: el 30% temporal se divide en 15% val y 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # Normalizacion Z-Score ajustada solo con train
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    def to_dataset(X_split, y_split):
        return TensorDataset(
            torch.tensor(X_split, dtype=torch.float32),
            torch.tensor(y_split, dtype=torch.long),
        )

    train_loader = DataLoader(to_dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(to_dataset(X_val, y_val),     batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(to_dataset(X_test, y_test),   batch_size=batch_size, shuffle=False)

    if verbose:
        print(f"Cortes de los terciles de {TARGET_COLUMN}: {np.round(bins, 2)}")
        print(f"Numero de features (n): {len(feature_names)}")
        print(f"Features: {feature_names}")
        print(f"Tamano del conjunto de entrenamiento: {len(train_loader.dataset)}")
        print(f"Tamano del conjunto de validacion: {len(val_loader.dataset)}")
        print(f"Tamano del conjunto de prueba: {len(test_loader.dataset)}")

    info = {
        "feature_names": feature_names,
        "input_dim": len(feature_names),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "bins": bins,
        "scaler": scaler,
    }

    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    get_dataloaders()
