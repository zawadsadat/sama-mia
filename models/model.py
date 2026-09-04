"""
Shared model architectures.
Used by: target model training, shadow models, FL clients.
"""

import torch.nn as nn


class TargetModel(nn.Module):
    """
    Binary classifier.
    Deliberately overparameterized to encourage memorization,
    which makes membership inference attacks more effective.
    """

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class AttackModel(nn.Module):
    """
    MIA attack classifier.
    Takes attack features (prob0, prob1, loss, entropy) and predicts
    member (1) vs non-member (0).
    """

    def __init__(self, input_dim=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
