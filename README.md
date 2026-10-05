# TorchGuard

**AI Model Security Testing**

TorchGuard is an early-stage security testing framework for identifying anomalies in AI models before deployment.

**Current prototype:** PyTorch model artifacts (`.pt` / `.pth`)

## Checks

* 🔐 Model integrity
* 🧱 Model structure
* 🔢 Weight anomalies (`NaN` / `Inf`)
* 🧠 Model behavior
* 📈 Gradient anomalies
* 🎯 Input sensitivity

## Quick Start

```bash
python -m pytest
python -m torchguard.scanner examples/model.pt
```

Test anomaly detection:

```bash
python -m torchguard.scanner examples/anomalous_model.pt
python -m torchguard.scanner examples/nan_model.pt
```

## Status

**v0.1 — Early prototype**

TorchGuard identifies security-relevant anomalies for further investigation. It does not determine whether a model is malicious or safe.

## Roadmap

* Broader model support
* Model baseline comparison
* Adversarial testing
* Backdoor detection
* CI/CD integration
* Black-box LLM testing

## License

Apache License 2.0

Copyright © 2026 Eyitayo Alimi
