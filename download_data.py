"""
Descarga del dataset de imagenes de productos de moda desde Kaggle.

Se usa la variante "small" (60x80 px, ~600 MB): contiene exactamente las
mismas 44.000 imagenes y el mismo styles.csv que la version full-res de
25 GB, pero a una resolucion que de todos modos es mayor que la que
necesita el MLP tras el reescalado.

Requiere credenciales de Kaggle en ~/.kaggle/kaggle.json
(Kaggle > Settings > API > Create New API Token).
"""

import kagglehub

DATASET = "paramaggarwal/fashion-product-images-small"


def download():
    path = kagglehub.dataset_download(DATASET)
    print("Path to dataset files:", path)
    return path


if __name__ == "__main__":
    download()
