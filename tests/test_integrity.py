from torchguard.integrity import calculate_sha256


def test_sha256():
    test_file = "tests/test_file.txt"

    with open(test_file, "w") as file:
        file.write("TorchGuard")

    result = calculate_sha256(test_file)

    assert len(result) == 64