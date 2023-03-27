import torch
import torch.nn as nn


class RelationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 64)
        self.linear2 = nn.Linear(64, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 80)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        # x = x - torch.amax(x, dim=1, keepdim=True)
        x = nn.functional.softmax(x, dim=1)
        return x

