import torch.nn as nn


class MLP(nn.Module):
    """
    Perceptron Multicapa para clasificacion de imagenes.

    La imagen se recibe ya aplanada como un vector de
    image_size * image_size * 3 componentes. La arquitectura sigue el
    mismo esquema del ejemplo de MNIST visto en clase (capas totalmente
    conectadas con ReLU), ampliado porque la entrada es mucho mayor y en
    color.

    Se agrega BatchNorm para estabilizar el entrenamiento con una
    entrada de alta dimension, y Dropout para contener el sobreajuste.
    """

    def __init__(self, input_dim, num_classes, hidden=(512, 256, 128), dropout=0.3):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden[0])
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.bn2 = nn.BatchNorm1d(hidden[1])
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden[1], hidden[2])
        self.bn3 = nn.BatchNorm1d(hidden[2])
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(hidden[2], num_classes)

    def forward(self, x):

        # Primera capa oculta
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        # Segunda capa oculta
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        # Tercera capa oculta
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        # Capa de salida (logits, sin softmax: lo aplica CrossEntropyLoss)
        x = self.fc4(x)

        return x
