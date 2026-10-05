import torch


clean_model = torch.load(
    "examples/model.pt",
    map_location="cpu",
    weights_only=True,
)

# Deliberately introduce an anomalous value.
clean_model["network.0.weight"][0, 0] = float("inf")

torch.save(
    clean_model,
    "examples/anomalous_model.pt",
)

print("Anomalous model saved to examples/anomalous_model.pt")