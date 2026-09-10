"""
Analisis exploratorio del dataset. Genera las figuras usadas en el informe.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from dataset import CLASS_NAMES, TARGET_COLUMN, build_target, load_dataframe

df = load_dataframe()

print("Dimensiones (m, n):", df.shape)
print("\nTipos de dato:")
print(df.dtypes)
print("\nValores nulos:", df.isna().sum().sum())
print("Filas duplicadas:", df.duplicated().sum())
print("\nEstadisticas descriptivas:")
print(df.describe().T)

for column in ["Gender", "Education_Level"]:
    print(f"\nDistribucion de {column}:")
    print(df[column].value_counts())

_, y, bins = build_target(df)

print(f"\nCortes de los terciles de {TARGET_COLUMN}: {bins.round(2)}")

# Distribucion de la clase objetivo
plt.figure(figsize=(6, 4))
sns.countplot(x=[CLASS_NAMES[label] for label in y], order=CLASS_NAMES)
plt.xlabel("Nivel de salud mental")
plt.ylabel("Numero de estudiantes")
plt.title("Distribucion de la variable objetivo")
plt.savefig("figures/eda_distribucion_clases.png", dpi=150, bbox_inches="tight")
plt.close()

# Histogramas de las variables numericas
numeric = df.select_dtypes(include="number")

numeric.hist(figsize=(12, 8), bins=40)
plt.tight_layout()
plt.savefig("figures/eda_histogramas.png", dpi=150, bbox_inches="tight")
plt.close()

# Matriz de correlacion
plt.figure(figsize=(8, 6))
sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlacion")
plt.savefig("figures/eda_correlacion.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nCorrelacion con el puntaje de salud mental:")
print(numeric.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values())

print("\nFiguras guardadas en figures/")
