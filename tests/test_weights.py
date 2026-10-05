import torch

from torchguard.weights import inspect_weights


def test_clean_model_has_no_findings():
    findings = inspect_weights("examples/model.pt")

    assert findings == []


def test_anomalous_model_is_detected():
    findings = inspect_weights("examples/anomalous_model.pt")

    assert len(findings) == 1
    assert findings[0].severity == "HIGH"
    assert findings[0].category == "WEIGHT_ANOMALY"
    assert "Inf values" in findings[0].message