import torch.nn as nn


class MLP(nn.Module):
    """
    Perceptron Multicapa para clasificacion multiclase sobre datos
    tabulares.

    A diferencia del ejemplo con MNIST, la entrada ya es un vector plano
    de caracteristicas, por lo que no se aplica ningun aplanamiento.
    Se agrega Dropout porque con pocas features y muchas muestras la red
    tiende a sobreajustar rapidamente.
    """

    def __init__(self, input_dim, num_classes, hidden=(64, 32), dropout=0.2):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden[0])
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden[1], num_classes)

    def forward(self, x):

        # Primera capa oculta
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        # Segunda capa oculta
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        # Capa de salida (logits, sin softmax: lo aplica CrossEntropyLoss)
        x = self.fc3(x)

        return x
