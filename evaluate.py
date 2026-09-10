"""
Evaluacion final sobre el conjunto de prueba.

El conjunto de test se usa UNA SOLA VEZ, con el mejor checkpoint
seleccionado por perdida de validacion durante el entrenamiento.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from dataset import get_dataloaders
from model import MLP

image_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)


def predict(model, loader):
    """Devuelve las etiquetas reales y las predichas para un DataLoader."""

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for inputs, targets in loader:

            inputs = inputs.to(device).float()

            logits = model(inputs)
            predictions = logits.argmax(dim=1)

            y_true.append(targets.numpy())
            y_pred.append(predictions.cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Matriz de confusion normalizada por fila (recall por clase)."""

    matrix = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(14, 12))

    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Proporcion de la clase real"},
    )

    plt.xlabel("Prediccion")
    plt.ylabel("Valor real")
    plt.title("Matriz de confusion normalizada - Conjunto de prueba")

    plt.savefig("figures/matriz_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_worst_classes(y_true, y_pred, class_names, top=10):
    """Grafica las clases con peor F1-Score."""

    scores = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)))

    order = np.argsort(scores)[:top]

    plt.figure(figsize=(8, 5))

    plt.barh(
        [class_names[i] for i in order][::-1],
        [scores[i] for i in order][::-1],
        color="indianred",
    )

    plt.xlabel("F1-Score")
    plt.xlim(0, 1)
    plt.title(f"Las {top} clases peor clasificadas")
    plt.grid(axis="x")

    plt.savefig("figures/peores_clases.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    _, _, test_loader, info = get_dataloaders(
        batch_size=128, image_size=image_size, verbose=False
    )

    class_names = info["class_names"]

    model = MLP(input_dim=info["input_dim"], num_classes=info["num_classes"])
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)

    y_true, y_pred = predict(model, test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\nAccuracy en el conjunto de prueba: {100 * accuracy:.2f}%")
    print(f"F1-Score macro: {macro_f1:.4f}")
    print(f"F1-Score ponderado: {weighted_f1:.4f}\n")

    print("Reporte de clasificacion:")
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, digits=4, zero_division=0
        )
    )

    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_worst_classes(y_true, y_pred, class_names)

    # Pares de clases que mas se confunden entre si
    matrix = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(matrix, 0)

    pairs = []

    for real in range(len(class_names)):
        for predicted in range(len(class_names)):
            if matrix[real, predicted] > 0:
                pairs.append((matrix[real, predicted], class_names[real], class_names[predicted]))

    pairs.sort(reverse=True)

    print("Confusiones mas frecuentes (real -> predicha):")

    for count, real, predicted in pairs[:15]:
        print(f"  {count:>4}  {real} -> {predicted}")

    print("\nFiguras guardadas en figures/")
