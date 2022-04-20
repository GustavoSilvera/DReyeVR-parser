import torch


class SteeringModel(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1  # outputting only a single scalar
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, 64),
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 128),
            torch.nn.Linear(128, 256),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, 128),
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, self.out_dim),
        )

    def forward(self, x):
        return self.network(x)
