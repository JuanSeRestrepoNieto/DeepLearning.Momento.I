"""
Carga y preprocesamiento del dataset
"Fashion Product Images (Small)" de Kaggle.

Tarea: clasificacion multiclase de la subcategoria del producto
(Topwear, Shoes, Bags, Watches, ...) a partir de la imagen.

Las imagenes se reescalan a IMAGE_SIZE x IMAGE_SIZE en RGB y se aplanan
en un vector, que es la entrada del Perceptron Multicapa.
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Resolucion a la que se reescala cada imagen antes de aplanarla.
# 32x32x3 = 3.072 features de entrada para el MLP.
IMAGE_SIZE = 32

TARGET_COLUMN = "subCategory"

# Las clases con muy pocas muestras se descartan: no permiten una
# particion estratificada en tres subconjuntos ni una evaluacion
# estadisticamente significativa.
MIN_SAMPLES_PER_CLASS = 100

RANDOM_STATE = 42

CACHE_PATH = "data/fashion_cache_{size}.npz"


def find_dataset_path():
    """
    Localiza la carpeta descargada por kagglehub. Si no existe, lanza un
    error indicando como obtenerla.
    """

    import kagglehub

    from download_data import DATASET

    try:
        return kagglehub.dataset_download(DATASET)
    except Exception as error:
        raise RuntimeError(
            "No se pudo obtener el dataset desde Kaggle. Verifica que "
            "exista ~/.kaggle/kaggle.json con tu token de API.\n"
            f"Error original: {error}"
        ) from error


def load_metadata(root):
    """
    Lee styles.csv y conserva unicamente las filas cuya imagen existe
    realmente en disco y cuya clase tiene suficientes muestras.
    """

    styles_path = os.path.join(root, "styles.csv")
    images_dir = os.path.join(root, "images")

    # Algunas filas de styles.csv tienen comas de mas en
    # productDisplayName, por eso se omiten las lineas malformadas.
    df = pd.read_csv(styles_path, on_bad_lines="skip")

    print(f"Filas en styles.csv: {len(df)}")

    df = df.dropna(subset=[TARGET_COLUMN])

    # Conservar solo las filas con imagen disponible
    df["path"] = df["id"].apply(lambda i: os.path.join(images_dir, f"{i}.jpg"))
    df = df[df["path"].apply(os.path.exists)]

    print(f"Filas con imagen disponible: {len(df)}")

    # Descartar clases minoritarias
    counts = df[TARGET_COLUMN].value_counts()
    rare = counts[counts < MIN_SAMPLES_PER_CLASS]

    if len(rare) > 0:
        print(
            f"Clases descartadas por tener menos de {MIN_SAMPLES_PER_CLASS} "
            f"muestras ({len(rare)}): {list(rare.index)}"
        )

    keep = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    df = df[df[TARGET_COLUMN].isin(keep)]

    df = df.reset_index(drop=True)

    print(f"Muestras finales (m): {len(df)}")
    print(f"Numero de clases: {df[TARGET_COLUMN].nunique()}")

    return df


def build_arrays(df, image_size=IMAGE_SIZE):
    """
    Decodifica cada imagen, la reescala y la almacena como uint8.
    Se guarda en uint8 (no float32) para que el arreglo completo ocupe
    cuatro veces menos memoria.
    """

    class_names = sorted(df[TARGET_COLUMN].unique())
    class_to_index = {name: index for index, name in enumerate(class_names)}

    X = np.zeros((len(df), image_size, image_size, 3), dtype=np.uint8)
    y = np.zeros(len(df), dtype=np.int64)

    for position, row in enumerate(df.itertuples(index=False)):

        with Image.open(row.path) as image:
            image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            X[position] = np.asarray(image, dtype=np.uint8)

        y[position] = class_to_index[getattr(row, TARGET_COLUMN)]

        if (position + 1) % 5000 == 0:
            print(f"  procesadas {position + 1}/{len(df)} imagenes")

    return X, y, class_names


def get_arrays(image_size=IMAGE_SIZE, use_cache=True):
    """Construye los arreglos de imagenes, con cache en disco."""

    cache_path = CACHE_PATH.format(size=image_size)

    if use_cache and os.path.exists(cache_path):
        print(f"Cargando cache: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        return cache["X"], cache["y"], list(cache["class_names"])

    root = find_dataset_path()
    print(f"Dataset en: {root}")

    df = load_metadata(root)

    print("Decodificando y reescalando imagenes...")
    X, y, class_names = build_arrays(df, image_size)

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, class_names=np.array(class_names))
    print(f"Cache guardada en {cache_path}")

    return X, y, class_names


class FashionDataset(Dataset):
    """
    Entrega cada imagen ya aplanada y normalizada como un vector.

    La normalizacion es Z-Score por canal, con la media y la desviacion
    calculadas UNICAMENTE sobre el conjunto de entrenamiento.
    """

    def __init__(self, X, y, mean, std):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        image = self.X[index].astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        # Aplanar a un vector: (H, W, C) -> (H * W * C)
        features = torch.from_numpy(image.reshape(-1))
        label = torch.tensor(self.y[index], dtype=torch.long)

        return features, label


def get_dataloaders(batch_size=128, image_size=IMAGE_SIZE, verbose=True):
    """
    Devuelve los DataLoaders de entrenamiento, validacion y prueba con
    particion estratificada 70 / 15 / 15.
    """

    X, y, class_names = get_arrays(image_size)

    # Primera particion: 70% train, 30% temporal
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    # Segunda particion: el 30% temporal se divide en 15% val y 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Estadisticas por canal calculadas solo con train (evita data leakage)
    train_float = X_train.astype(np.float32) / 255.0

    mean = train_float.mean(axis=(0, 1, 2))
    std = train_float.std(axis=(0, 1, 2))

    del train_float

    train_loader = DataLoader(
        FashionDataset(X_train, y_train, mean, std), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        FashionDataset(X_val, y_val, mean, std), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        FashionDataset(X_test, y_test, mean, std), batch_size=batch_size, shuffle=False
    )

    # Pesos por clase: el dataset esta fuertemente desbalanceado
    counts = np.bincount(y_train, minlength=len(class_names))
    class_weights = counts.sum() / (len(class_names) * np.maximum(counts, 1))

    if verbose:
        print(f"\nResolucion: {image_size}x{image_size}x3")
        print(f"Numero de features (n): {image_size * image_size * 3}")
        print(f"Numero de clases: {len(class_names)}")
        print(f"Tamano del conjunto de entrenamiento: {len(y_train)}")
        print(f"Tamano del conjunto de validacion: {len(y_val)}")
        print(f"Tamano del conjunto de prueba: {len(y_test)}")

    info = {
        "input_dim": image_size * image_size * 3,
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "class_weights": torch.tensor(class_weights, dtype=torch.float32),
        "train_counts": counts,
    }

    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    get_dataloaders()
