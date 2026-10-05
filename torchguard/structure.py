import torch


def inspect_model_structure(model_path: str) -> None:
    """Inspect the structure of a PyTorch state-dict model."""

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    print("\nMODEL STRUCTURE")
    print("===============")

    total_parameters = 0

    for name, tensor in state_dict.items():
        parameter_count = tensor.numel()
        total_parameters += parameter_count

        print(
            f"[INFO] {name}: "
            f"shape={tuple(tensor.shape)}, "
            f"parameters={parameter_count}"
        )

    print(f"[INFO] Total parameters: {total_parameters}")