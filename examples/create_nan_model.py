import torch


clean_model = torch.load(
    "examples/model.pt",
    map_location="cpu",
    weights_only=True,
)

# Deliberately introduce a NaN value.
clean_model["network.0.weight"][0, 0] = float("nan")

torch.save(
    clean_model,
    "examples/nan_model.pt",
)

print("NaN model saved to examples/nan_model.pt")