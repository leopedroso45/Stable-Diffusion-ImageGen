from sevsd.setup_device import setup_device
from diffusers import StableDiffusionPipeline

def setup_pipeline(config):
    device = setup_device()
    pretrained_model_link_or_path, cache_dir = config
    pipeline = None
    if pretrained_model_link_or_path.endswith(".safetensors"):
        pipeline = StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path,
            cache_dir=cache_dir,
            use_safetensors=True,
            load_safety_checker=False,
            requires_safety_checker=False
        ).to(device)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_link_or_path,
            cache_dir=cache_dir,
            #use_safetensors=False,
            load_safety_checker=False,
            requires_safety_checker=False,
        ).to(device)

    pipeline.enable_attention_slicing()
    return pipeline