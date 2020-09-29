import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCov(nn.Module):
    def __init__(self):
        super().__init__()
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


class SimpleFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(75, 225)
        self.fc2 = nn.Linear(225, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_optim(net: nn.Module, lr):
    return torch.optim.Adam(net.parameters(), lr)


class FullNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11644, 5822)
        self.fc2 = nn.Linear(5822, 2911)
        self.fc3 = nn.Linear(2911, 1455)
        self.fc4 = nn.Linear(1455, 800)
        self.fc5 = nn.Linear(800, 400)
        self.fc6 = nn.Linear(400, 200)
        self.fc7 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


class LargeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11644, 7762)
        self.fc2 = nn.Linear(7762, 5174)
        self.fc3 = nn.Linear(5174, 3449)
        self.fc4 = nn.Linear(3449, 2229)
        self.fc5 = nn.Linear(2229, 1532)
        self.fc6 = nn.Linear(1532, 1021)
        self.fc7 = nn.Linear(1021, 500)
        self.fc8 = nn.Linear(500, 250)
        self.fc9 = nn.Linear(250, 125)
        self.fc10 = nn.Linear(125, 10)
        self.fc11 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        return x


class SmallFullNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11644, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc4 = nn.Linear(1000, 400)
        self.fc5 = nn.Linear(400, 100)
        self.fc6 = nn.Linear(100, 50)
        self.fc7 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


class SmallFullNetWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11644, 2911)
        self.d1 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(2911, 800)
        self.d2 = nn.Dropout(p=0.2)
        self.fc5 = nn.Linear(800, 400)
        self.fc6 = nn.Linear(400, 200)
        self.fc7 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc4(x))
        self.d2(x)
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x
