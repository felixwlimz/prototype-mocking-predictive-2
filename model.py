import torch
import torch.nn as nn


class PredictorNet(nn.Module):
    def __init__(self, input_dim):
        super(PredictorNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            # Output Layer
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x)

