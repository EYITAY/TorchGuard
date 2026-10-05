import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


model = SimpleClassifier()

torch.save(model.state_dict(), "examples/model.pt")

print("Model saved to examples/model.pt")