import torch
from datetime import datetime
from tasks.tasks import tasks
from diffusers import StableDiffusionPipeline
from pipeline_config.config import config_1 as configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")

def get_config(config):
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

def generate_image(args, pipeline):
    prompt, negative_prompt, num_inference_steps, num_images, cfg = args
    try:
        with torch.no_grad():
            output = pipeline(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, num_images_per_prompt=num_images, guidance_scale=cfg)
            print(f"Available keys in output: {output.keys()}\nNumber of items inside the output is: {len(output['images'])}")
            return output["images"]
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        return None

def process_task(tasks, pipeline):
    try:
        if tasks is not None:
            for task in tasks:
                images = generate_image(task, pipeline)
                if images is not None:
                    for image in images:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                        image_path = f"./generated-images2/generated_image_{timestamp}.png"
                        image.save(image_path)
                        print(f"Image saved at {image_path}")
                else:
                    print("Image generation failed due to memory constraints.")
    except Exception as e:
        print(f"Exception: {e}")
        return None

def do_work():
    for config in configs:
        pipeline = get_config(config)
        process_task(tasks, pipeline)

do_work()
