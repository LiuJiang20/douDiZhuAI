import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCov(nn.Module):
    def __init__(self):
        super(SimpleCov, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv1d(5, 128, 3)
        self.conv2 = nn.Conv1d(64, 64, 3)
        # an affine operation: y = Wx + b
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 120)  # 5*15 from info dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_optim(net: nn.Module, lr):
    return torch.optim.Adam(net.parameters(), lr)
