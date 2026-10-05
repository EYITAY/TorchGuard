import torch
import torch.nn as nn

from .report import SecurityFinding


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.network(x)


def inspect_behavior(model_path: str) -> list[SecurityFinding]:
    """Run basic behavioral security checks on a PyTorch model."""

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model = SimpleClassifier()
    model.load_state_dict(state_dict)
    model.eval()

    test_inputs = {
        "zeros": torch.zeros(1, 10),
        "ones": torch.ones(1, 10),
        "random": torch.randn(1, 10),
    }

    findings = []

    print("\nBEHAVIOR INSPECTION")
    print("===================")

    for name, test_input in test_inputs.items():

        with torch.no_grad():
            output = model(test_input)

        if torch.isfinite(output).all():
            print(f"[PASS] {name}: finite output")
        else:
            print(f"[WARN] {name}: NaN or Inf output")

            findings.append(
                SecurityFinding(
                    severity="HIGH",
                    category="BEHAVIOR_ANOMALY",
                    message=f"{name} input produced NaN or Inf output",
                )
            )

        print(f"       input shape={tuple(test_input.shape)}")
        print(f"       output shape={tuple(output.shape)}")
        print(f"       output={output.tolist()}")

    return findings