"""
Analisis exploratorio del dataset de imagenes de productos de moda.
Genera las figuras usadas en el informe.
"""

import matplotlib.pyplot as plt
import numpy as np

from dataset import IMAGE_SIZE, TARGET_COLUMN, get_arrays

X, y, class_names = get_arrays(IMAGE_SIZE)

print(f"\nDimensiones del tensor de imagenes: {X.shape}")
print(f"Muestras (m): {len(y)}")
print(f"Features tras aplanar (n): {X.shape[1] * X.shape[2] * X.shape[3]}")
print(f"Numero de clases: {len(class_names)}")

counts = np.bincount(y, minlength=len(class_names))
order = np.argsort(counts)[::-1]

print(f"\nDistribucion de {TARGET_COLUMN}:")

for index in order:
    print(f"  {class_names[index]:<20} {counts[index]:>6}  ({100 * counts[index] / len(y):.2f}%)")

print(f"\nClase mayoritaria: {class_names[order[0]]} ({counts[order[0]]} muestras)")
print(f"Clase minoritaria: {class_names[order[-1]]} ({counts[order[-1]]} muestras)")
print(f"Razon de desbalance: {counts[order[0]] / counts[order[-1]]:.1f} a 1")

# Distribucion de clases
plt.figure(figsize=(10, 8))

plt.barh(
    [class_names[i] for i in order][::-1],
    [counts[i] for i in order][::-1],
    color="steelblue",
)

plt.xlabel("Numero de imagenes")
plt.title(f"Distribucion de la variable objetivo ({TARGET_COLUMN})")
plt.grid(axis="x")

plt.savefig("figures/eda_distribucion_clases.png", dpi=150, bbox_inches="tight")
plt.close()

# Muestra de imagenes, una por clase
sample_classes = order[:16]

figure, axes = plt.subplots(4, 4, figsize=(10, 10))

for axis, class_index in zip(axes.ravel(), sample_classes):

    position = np.where(y == class_index)[0][0]

    axis.imshow(X[position])
    axis.set_title(class_names[class_index], fontsize=9)
    axis.axis("off")

plt.suptitle(f"Ejemplos por clase (reescalados a {IMAGE_SIZE}x{IMAGE_SIZE})")
plt.tight_layout()

plt.savefig("figures/eda_ejemplos.png", dpi=150, bbox_inches="tight")
plt.close()

# Imagen promedio de las clases mas frecuentes: muestra que tanta
# estructura espacial comun hay dentro de cada clase.
figure, axes = plt.subplots(2, 4, figsize=(12, 6))

for axis, class_index in zip(axes.ravel(), order[:8]):

    mean_image = X[y == class_index].mean(axis=0).astype(np.uint8)

    axis.imshow(mean_image)
    axis.set_title(class_names[class_index], fontsize=9)
    axis.axis("off")

plt.suptitle("Imagen promedio por clase")
plt.tight_layout()

plt.savefig("figures/eda_imagen_promedio.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nFiguras guardadas en figures/")
