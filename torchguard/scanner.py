import sys

from .behavior import inspect_behavior
from .gradients import inspect_gradients
from .integrity import calculate_sha256
from .robustness import inspect_robustness
from .structure import inspect_model_structure
from .weights import inspect_weights


def scan_model(model_path: str) -> None:
    """Run a basic security scan on a PyTorch model."""

    print("TorchGuard Security Scanner")
    print("===========================")

    all_findings = []

    # Integrity check
    file_hash = calculate_sha256(model_path)

    print("[PASS] Model file readable")
    print(f"[INFO] SHA-256: {file_hash}")

    # Structure check
    inspect_model_structure(model_path)

    # Weight inspection
    weight_findings = inspect_weights(model_path)
    all_findings.extend(weight_findings)

    # Behavioral inspection
    behavior_findings = inspect_behavior(model_path)
    all_findings.extend(behavior_findings)

    # Gradient inspection
    gradient_findings = inspect_gradients(model_path)
    all_findings.extend(gradient_findings)

    # Robustness / perturbation inspection
    robustness_findings = inspect_robustness(model_path)
    all_findings.extend(robustness_findings)

    # Security summary
    print("\nSECURITY SUMMARY")
    print("================")

    if not all_findings:
        print("[PASS] No security findings detected")
        print("[INFO] Risk level: LOW")
    else:
        print(f"[WARN] Findings detected: {len(all_findings)}")

        highest_severity = "LOW"

        for finding in all_findings:
            print(f"[{finding.severity}] {finding.category}")
            print(f"       {finding.message}")

            if finding.severity == "HIGH":
                highest_severity = "HIGH"
            elif (
                finding.severity == "MEDIUM"
                and highest_severity == "LOW"
            ):
                highest_severity = "MEDIUM"

        print(f"[WARN] Risk level: {highest_severity}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m torchguard.scanner <model_path>")
        sys.exit(1)

    scan_model(sys.argv[1])