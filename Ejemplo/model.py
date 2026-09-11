import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 10)



    def forward(self, x):

        x = x.view(x.size(0), -1)

        # Primera capa
        x = self.fc1(x)
        x = self.relu1(x)

        # Segunda capa
        x = self.fc2(x)
        x = self.relu2(x)

        # Capa de salida
        x = self.fc3(x)

        return x