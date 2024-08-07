from sevsd.setup_device import setup_device
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from transformers import AutoFeatureExtractor
import os

def setup_pipeline(pretrained_model_link_or_path, loras, positive_embeddings=None, negative_embeddings=None, **kwargs):
    r"""
    Sets up and returns a Stable Diffusion pipeline for image generation.

    This function initializes the Stable Diffusion pipeline using either a pretrained model link or a local file path. It automatically determines the appropriate device (CPU or GPU) for running the model and applies necessary configuration parameters.

    Parameters:
        pretrained_model_link_or_path (str): A link to a pretrained model or a file path to a local model file.
        loras (list): A list of LoRA weights files to be applied to the pipeline.
        positive_embeddings (list, optional): A list of positive embeddings to be applied to the pipeline. Defaults to None.
        negative_embeddings (list, optional): A list of negative embeddings to be applied to the pipeline. Defaults to None.
        **kwargs: Additional keyword arguments for pipeline configuration.

    Returns:
        StableDiffusionPipeline: The initialized Stable Diffusion pipeline ready for image generation.

    Example:
        pipeline = setup_pipeline("CompVis/stable-diffusion-v1-4", ["./loras/lora1.safetensors", "./loras/lora2.safetensors"])

    Note:
        - The function supports both remote model links and local `.safetensors` files.
        - It automatically disables the safety checker for faster inference unless specified otherwise in `**kwargs`.
        - The pipeline is configured to use the most efficient device available (CUDA, MPS, or CPU).
    """

    device = setup_device()

    default_kwargs = {
        "use_safetensors": False,
        "safety_checker": None,
    }

    if pretrained_model_link_or_path.endswith(".safetensors"):
        default_kwargs["use_safetensors"] = True
        default_kwargs.update(kwargs)

        pipeline = StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path,
            **default_kwargs
        )
    else:
        default_kwargs["feature_extractor"] = AutoFeatureExtractor.from_pretrained(pretrained_model_link_or_path)
        default_kwargs.update(kwargs)
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_link_or_path,
            **default_kwargs
        )
    
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

    if loras:
        pipeline.unfuse_lora()
        
        set_loras = []
        set_weights = []
        for lora in loras:
            adapter_name = os.path.basename(lora).replace(".", "")
            pipeline.load_lora_weights(
                lora,
                weight_name=lora,
                adapter_name=adapter_name
            )
            set_loras.append(adapter_name)
            set_weights.append(1.0)

        pipeline.set_adapters(set_loras, set_weights)
        pipeline.fuse_lora()

    pipeline.positive_embeddings = positive_embeddings
    pipeline.negative_embeddings = negative_embeddings

    pipeline.to(device)
    pipeline.enable_attention_slicing()

    return pipeline