import torch

def test_gpu_availability():
    """
    Test to check if a GPU is available for PyTorch.
    """
    assert torch.cuda.is_available(), "GPU is not available"
    print("GPU is available for PyTorch.")

test_gpu_availability()