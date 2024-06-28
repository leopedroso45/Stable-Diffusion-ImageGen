from sevsd.setup_device import setup_device
from diffusers import StableDiffusionPipeline
from transformers import AutoFeatureExtractor

def setup_pipeline(pretrained_model_link_or_path, **kwargs):
    r"""
    Sets up and returns a Stable Diffusion pipeline for image generation.

    This function initializes the Stable Diffusion pipeline using either a pretrained model link or a local file path. It automatically determines the appropriate device (CPU or GPU) for running the model and applies necessary configuration parameters.

    Parameters:
        pretrained_model_link_or_path (str): A link to a pretrained model or a file path to a local model file.
        **kwargs: Additional keyword arguments for pipeline configuration.

    Returns:
        StableDiffusionPipeline: The initialized Stable Diffusion pipeline ready for image generation.

    Example:
        pipeline = setup_pipeline("CompVis/stable-diffusion-v1-4")

    Note:
        - The function supports both remote model links and local `.safetensors` files.
        - It automatically disables the safety checker for faster inference unless specified otherwise in `**kwargs`.
        - The pipeline is configured to use the most efficient device available (CUDA, MPS, or CPU).
    """

    device = setup_device()

    default_kwargs = {
        "use_safetensors": False,
        "safety_checker": None,
        "feature_extractor": AutoFeatureExtractor.from_pretrained(pretrained_model_link_or_path),
    }

    if pretrained_model_link_or_path.endswith(".safetensors"):
        default_kwargs["use_safetensors"] = True
        default_kwargs.update(kwargs)

        pipeline = StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path,
            **default_kwargs
        )
    else:
        default_kwargs.update(kwargs)
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_link_or_path,
            **default_kwargs
        )

    pipeline.to(device)
    pipeline.enable_attention_slicing()

    return pipeline