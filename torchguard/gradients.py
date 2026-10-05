import torch
import torch.nn as nn

from .behavior import SimpleClassifier
from .report import SecurityFinding


def inspect_gradients(model_path: str) -> list[SecurityFinding]:
    """Inspect model gradients for numerical anomalies."""

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model = SimpleClassifier()
    model.load_state_dict(state_dict)
    model.train()

    test_input = torch.randn(1, 10)

    output = model(test_input)

    loss = output.sum()

    loss.backward()

    findings = []

    print("\nGRADIENT INSPECTION")
    print("===================")

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            print(f"[WARN] {name}: no gradient")
            continue

        gradient = parameter.grad

        has_nan = torch.isnan(gradient).any().item()
        has_inf = torch.isinf(gradient).any().item()
        max_abs = gradient.abs().max().item()

        if has_nan:
            print(f"[WARN] {name}: gradient contains NaN")

            findings.append(
                SecurityFinding(
                    severity="HIGH",
                    category="GRADIENT_ANOMALY",
                    message=f"{name} gradient contains NaN values",
                )
            )

        elif has_inf:
            print(f"[WARN] {name}: gradient contains Inf")

            findings.append(
                SecurityFinding(
                    severity="HIGH",
                    category="GRADIENT_ANOMALY",
                    message=f"{name} gradient contains Inf values",
                )
            )

        else:
            print(f"[PASS] {name}: gradient is finite")

        print(f"       max_abs={max_abs:.6f}")

    return findings