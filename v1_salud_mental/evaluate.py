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
)

from dataset import get_dataloaders
from model import MLP

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


def permutation_importance(model, loader, baseline_accuracy, feature_names, repeats=5):
    """
    Importancia por permutacion: mide cuanto cae el accuracy al barajar
    aleatoriamente una feature. Entre mas cae, mas importante es.
    """

    X = loader.dataset.tensors[0].numpy()
    y = loader.dataset.tensors[1].numpy()

    rng = np.random.default_rng(42)
    importances = []

    model.eval()

    for index in range(X.shape[1]):

        drops = []

        for _ in range(repeats):

            X_permuted = X.copy()
            rng.shuffle(X_permuted[:, index])

            with torch.no_grad():
                logits = model(torch.tensor(X_permuted, dtype=torch.float32).to(device))
                predictions = logits.argmax(dim=1).cpu().numpy()

            drops.append(baseline_accuracy - accuracy_score(y, predictions))

        importances.append(np.mean(drops))

    order = np.argsort(importances)[::-1]

    return [(feature_names[i], importances[i]) for i in order]


if __name__ == "__main__":

    _, _, test_loader, info = get_dataloaders(batch_size=64, verbose=False)

    class_names = info["class_names"]

    model = MLP(input_dim=info["input_dim"], num_classes=info["num_classes"])
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)

    y_true, y_pred = predict(model, test_loader)

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nAccuracy en el conjunto de prueba: {100 * accuracy:.2f}%\n")

    print("Reporte de clasificacion:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Matriz de confusion
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )

    plt.xlabel("Prediccion")
    plt.ylabel("Valor real")
    plt.title("Matriz de confusion - Conjunto de prueba")

    plt.savefig("figures/matriz_confusion.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Importancia de las variables
    print("Importancia por permutacion (caida de accuracy):")

    for name, importance in permutation_importance(
        model, test_loader, accuracy, info["feature_names"]
    ):
        print(f"  {name:<32} {importance:+.4f}")
