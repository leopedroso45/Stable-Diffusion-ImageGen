import torch

def setup_device():
    r"""
    Sets up and returns the appropriate device for computation based on hardware availability.

    This function checks for the availability of CUDA (GPU) and MPS (Apple Silicon GPU) and accordingly sets the device. If neither CUDA nor MPS is available, it defaults to the CPU.

    Returns:
        torch.device: The device object representing either CUDA, MPS, or CPU.

    Example:
        device = setup_device()
        print(f"Using device: {device}")

    Note:
        This function prints a message indicating which device is being used. It helps optimize computations by utilizing available hardware acceleration.
    """
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
