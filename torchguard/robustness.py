import torch

from .behavior import SimpleClassifier
from .report import SecurityFinding


def inspect_robustness(model_path: str) -> list[SecurityFinding]:
    """Measure model sensitivity to a small input perturbation."""

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model = SimpleClassifier()
    model.load_state_dict(state_dict)
    model.eval()

    # Create a baseline input.
    base_input = torch.randn(1, 10)

    # Create a small controlled perturbation.
    perturbation = torch.full((1, 10), 0.001)

    perturbed_input = base_input + perturbation

    with torch.no_grad():
        base_output = model(base_input)
        perturbed_output = model(perturbed_input)

    # Measure how much the output changed.
    output_difference = torch.norm(
        perturbed_output - base_output
    ).item()

    findings = []

    print("\nROBUSTNESS INSPECTION")
    print("=====================")

    print(f"[INFO] Perturbation magnitude: {torch.norm(perturbation).item():.6f}")
    print(f"[INFO] Output change: {output_difference:.6f}")

    if torch.isfinite(perturbed_output).all():
        print("[PASS] Perturbed input produced finite output")
    else:
        print("[WARN] Perturbed input produced NaN or Inf")

        findings.append(
            SecurityFinding(
                severity="HIGH",
                category="ROBUSTNESS_ANOMALY",
                message="Perturbed input produced NaN or Inf output",
            )
        )

    return findings