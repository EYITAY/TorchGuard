import torch

from .report import SecurityFinding


def inspect_weights(model_path: str) -> list[SecurityFinding]:
    """Inspect model weights for numerical anomalies."""

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    findings = []

    print("\nWEIGHT INSPECTION")
    print("=================")

    for name, tensor in state_dict.items():
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        max_abs = tensor.abs().max().item()
        mean = tensor.mean().item()
        std = tensor.std().item()

        if has_nan:
            finding = SecurityFinding(
                severity="HIGH",
                category="WEIGHT_ANOMALY",
                message=f"{name} contains NaN values",
            )

            findings.append(finding)

            print(f"[WARN] {name}: contains NaN values")

        elif has_inf:
            finding = SecurityFinding(
                severity="HIGH",
                category="WEIGHT_ANOMALY",
                message=f"{name} contains Inf values",
            )

            findings.append(finding)

            print(f"[WARN] {name}: contains Inf values")

        else:
            print(f"[PASS] {name}: numerical values valid")

        print(f"       max_abs={max_abs:.6f}")
        print(f"       mean={mean:.6f}")
        print(f"       std={std:.6f}")

    return findings