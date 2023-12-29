from sevsd.setup_device import setup_device
from diffusers import StableDiffusionPipeline

def setup_pipeline(config, **kwargs):
    device = setup_device()
    pretrained_model_link_or_path, cache_dir = config

    default_kwargs = {
        "use_safetensors": False,
        "load_safety_checker": False,
        "requires_safety_checker": False,
        "cache_dir": cache_dir,
    }

    default_kwargs.update(kwargs)

    if pretrained_model_link_or_path.endswith(".safetensors"):
        default_kwargs["use_safetensors"] = True
        pipeline = StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path,
            **default_kwargs
        )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_link_or_path,
            **default_kwargs
        )

    pipeline.to(device)
    pipeline.enable_attention_slicing()

    return pipeline