import torch

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
