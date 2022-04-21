import torch


class SteeringModel(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1  # outputting only a single scalar
        layers = [
            torch.nn.Linear(self.in_dim, 64),
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 256),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, self.out_dim),
        ]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ThrottleModel(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1  # outputting only a single scalar
        layers = [
            torch.nn.Linear(self.in_dim, 128),
            torch.nn.Linear(128, 256),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),  # only positive
            torch.nn.Linear(256, self.out_dim),
        ]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        # throttle should be always positive
        return self.network(x)
