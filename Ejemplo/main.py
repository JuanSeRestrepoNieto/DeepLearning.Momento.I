import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MLP
from matplotlib import pyplot as plt

# Hiperparametros
batch_size  = 64
epochs      = 15
lr          = 0.0001
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir las transformaciones para los datos de entrenamiento y prueba
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Cargar el conjunto de datos MNIST
train_dataset   = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset    = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Obtener dataset de validacion
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000], generator=generator)

# Crear los DataLoaders para entrenamiento, validación y prueba
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Mostrar cantidad de datos en cada conjunto
print(f'Tamaño del conjunto de entrenamiento: {len(train_loader.dataset)}')
print(f'Tamaño del conjunto de validación: {len(val_loader.dataset)}')
print(f'Tamaño del conjunto de prueba: {len(test_loader.dataset)}')


# Verificar carda de un mini-batch de datos
images, labels = next(iter(train_loader))

print("\nDimensiones de un mini-batch:")

print("Imágenes:", images.shape)
print("Etiquetas:", labels.shape)

# Inicializar el modelo
model = MLP()
model.to(device)
print(model)

# Compilar el modelo
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


train_losses        = []
val_losses          = []
train_accuracies    = []
val_accuracies      = []

for epoch in range(epochs):
    
    model.train()
    running_loss    = 0.0
    correct         = 0
    total           = 0

    for batch_idx, data in enumerate(train_loader):

        # Obtener datos
        inputs, targets = data

        # Mover al dispositivo
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()

        # Reiniciar gradientes
        optimizer.zero_grad()

        # Forward propagation
        logits = model(inputs)

        # Calcular pérdida
        loss = criterion(logits, targets)

        # Backpropagation
        loss.backward()

        # Actualizar parámetros
        optimizer.step()
        
        running_loss += loss.item()
        
        # Predicciones
        predictions = logits.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
        
    running_val_loss    = 0.0
    correct             = 0
    total               = 0

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(device)
            targets = targets.to(device)

            # Forward propagation
            logits = model(images)

            # Calcular pérdida
            loss = criterion(logits, targets)

            running_val_loss += loss.item()

            # Predicciones
            predictions = logits.argmax(dim=1)

            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )




# ----------------------------------------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    range(1, epochs + 1),
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Época")
plt.ylim(0, 1)
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")

plt.legend()
plt.grid()

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    range(1, epochs + 1),
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Época")
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy vs Validation Accuracy")

plt.legend()
plt.grid()

plt.show()