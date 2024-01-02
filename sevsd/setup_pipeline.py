from sevsd.setup_device import setup_device
from diffusers import StableDiffusionPipeline

def setup_pipeline(pretrained_model_link_or_path, **kwargs):
    device = setup_device()

    default_kwargs = {
        "use_safetensors": False,
        "load_safety_checker": False,
        "requires_safety_checker": False,
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