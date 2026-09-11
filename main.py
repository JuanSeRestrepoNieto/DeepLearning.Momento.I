"""
Entrenamiento del MLP para clasificar la subcategoria de productos de
moda a partir de su imagen.

Momento Evaluativo I - Clasificacion con Redes Neuronales Profundas.
"""

import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

from dataset import get_dataloaders
from model import MLP

# Hiperparametros
batch_size  = 128
epochs      = 70
lr          = 0.001
patience    = 12
image_size  = 32
use_class_weights = True
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

# Cargar los datos (decodificacion, reescalado, split estratificado, Z-Score)
train_loader, val_loader, test_loader, info = get_dataloaders(
    batch_size=batch_size, image_size=image_size
)

# Verificar la carga de un mini-batch de datos
features, labels = next(iter(train_loader))

print("\nDimensiones de un mini-batch:")
print("Features:", features.shape)
print("Etiquetas:", labels.shape)

# Inicializar el modelo
model = MLP(input_dim=info["input_dim"], num_classes=info["num_classes"])
model.to(device)
print(model)

# Compilar el modelo. Se ponderan las clases porque el dataset esta
# fuertemente desbalanceado (Topwear concentra buena parte de las muestras).
weights = info["class_weights"].to(device) if use_class_weights else None

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses     = []
val_losses       = []
train_accuracies = []
val_accuracies   = []
val_macro_f1s    = []

# El checkpoint se elige por F1-Score macro de validacion y no por la
# perdida: al ponderar las clases, la perdida de validacion queda muy
# ruidosa y no refleja el desempeno sobre las clases minoritarias, que
# es justamente lo que interesa en un dataset con desbalance 148 a 1.
best_val_f1 = -1.0
epochs_without_improvement = 0
best_epoch = 0

for epoch in range(epochs):

    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for inputs, targets in train_loader:

        # Mover al dispositivo
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()

        # Reiniciar gradientes
        optimizer.zero_grad()

        # Forward propagation
        logits = model(inputs)

        # Calcular perdida
        loss = criterion(logits, targets)

        # Backpropagation
        loss.backward()

        # Actualizar parametros
        optimizer.step()

        running_loss += loss.item()

        # Predicciones
        predictions = logits.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    model.eval()
    running_val_loss = 0.0
    correct          = 0
    total            = 0

    val_true = []
    val_pred = []

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(device).float()
            targets = targets.to(device).long()

            # Forward propagation
            logits = model(inputs)

            # Calcular perdida
            loss = criterion(logits, targets)

            running_val_loss += loss.item()

            # Predicciones
            predictions = logits.argmax(dim=1)

            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            val_true.append(targets.cpu().numpy())
            val_pred.append(predictions.cpu().numpy())

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    val_macro_f1 = f1_score(
        np.concatenate(val_true), np.concatenate(val_pred), average="macro", zero_division=0
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    val_macro_f1s.append(val_macro_f1)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}% "
        f"Val F1: {val_macro_f1:.4f}"
    )

    # Early stopping: se guarda el modelo con mayor F1-Score macro
    if val_macro_f1 > best_val_f1:

        best_val_f1 = val_macro_f1
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(model.state_dict(), "best_model.pt")

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping en la epoca {epoch + 1}. "
                f"Mejor epoca: {best_epoch} (Val F1 macro: {best_val_f1:.4f})"
            )
            break

print(f"\nMejor modelo guardado en 'best_model.pt' (epoca {best_epoch}, "
      f"Val F1 macro: {best_val_f1:.4f}).")

# ----------------------------------------------------------------------------------------------

epochs_ran = len(train_losses)

plt.figure(figsize=(8, 5))

plt.plot(range(1, epochs_ran + 1), train_losses, marker="o", label="Training Loss")
plt.plot(range(1, epochs_ran + 1), val_losses, marker="o", label="Validation Loss")

plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")

plt.legend()
plt.grid()
plt.savefig("figures/curva_loss.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(range(1, epochs_ran + 1), train_accuracies, marker="o", label="Training Accuracy")
plt.plot(range(1, epochs_ran + 1), val_accuracies, marker="o", label="Validation Accuracy")

plt.xlabel("Epoca")
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy vs Validation Accuracy")

plt.legend()
plt.grid()
plt.savefig("figures/curva_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(range(1, epochs_ran + 1), val_macro_f1s, marker="o", color="darkgreen",
         label="Validation Macro F1")

plt.axvline(best_epoch, linestyle="--", color="gray", label=f"Mejor epoca ({best_epoch})")

plt.xlabel("Epoca")
plt.ylim(0, 1)
plt.ylabel("F1-Score macro")
plt.title("F1-Score macro en validacion")

plt.legend()
plt.grid()
plt.savefig("figures/curva_f1.png", dpi=150, bbox_inches="tight")
plt.show()
