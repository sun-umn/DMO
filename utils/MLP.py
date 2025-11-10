import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, num_layers=10, output_dim=2):
        super().__init__()
        bias = False
        layers = [nn.Linear(input_dim, hidden_dim, bias=bias)]
        for _ in range(num_layers-2):
            layers.append(nn.BatchNorm1d(num_features = hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        self.features = nn.Sequential(*layers)

        self.fc_layers = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, X):
        x = self.features(X)
        return self.fc_layers(x)
    
class Linear(nn.Module):
    def __init__(self, input_dim, output_dim=2, bias=True):
        super().__init__()
        self.fc_layer = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, X):
        return self.fc_layers(X)