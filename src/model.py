import torch
from torch import nn
from sklearn.linear_model import LogisticRegression

def make_baseline_model(random_state: int = 42):
    """
    Returns a logistic‐regression model to serve as our baseline.
    """
    # max_iter=1000 ensures convergence on our (3,800×8) data matrix
    return LogisticRegression(max_iter=1000, random_state=random_state)


class SimpleHabitabilityNet(nn.Module):
    """
    A small feed-forward network to predict habitability from 8 features.
    Architecture:
      • Input → Linear(8 → 32) → ReLU
      • → Dropout(0.2) → Linear(32 → 16) → ReLU
      • → Dropout(0.2) → Linear(16 → 1) → output logit
    We squeeze the output to a 1-D tensor so that each batch element maps to one logit.
    """
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 8) → output: (batch_size,)
        return self.net(x).squeeze(-1)