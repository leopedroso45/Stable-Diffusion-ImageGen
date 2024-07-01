import numpy as np
import torch
from PIL import Image
from RealESRGAN import RealESRGAN
from sevsd.setup_device import setup_device

device = setup_device()

def generate_image(job, pipeline, executor, upscaling_factor=4, parallel_exec=True, **kwargs):
    r"""
    Generates high-resolution images based on textual prompts using the provided Stable Diffusion pipeline and an upscaling model.

    This function handles the image generation process using specified parameters and execution configurations. 
    It supports both parallel and sequential execution modes for image generation tasks.

    Parameters:
        job (dict): A dictionary containing the 'prompt' and optionally 'negative_prompt' for image generation.
                    - 'prompt' (str): The textual prompt based on which the image will be generated.
                    - 'negative_prompt' (str, optional): A textual prompt that describes undesired features in the generated image.
        pipeline (callable): The Stable Diffusion pipeline callable used for generating images.
        executor (dict): A dictionary containing execution parameters for the pipeline.
                         - 'inference_steps' (int): Number of inference steps for the pipeline.
                         - 'num_of_exec' (int): Number of images to generate.
                         - 'cfg_scale' (float): The guidance scale for controlling image generation.
        upscaling_factor (int, optional): The factor by which to upscale the generated images. Defaults to 4.
        parallel_exec (bool, optional): If True, generates 'num_of_exec' images in parallel. Otherwise, generates images sequentially. Defaults to True.
        **kwargs: Additional keyword arguments passed to the pipeline function.

    Returns:
        list or None: A list of generated high-resolution images if successful, or None if a RuntimeError occurs during image generation.

    Raises:
        RuntimeError: If an error occurs during the image generation process.

    Example:
        >>> job = {"prompt": "A beautiful landscape", "negative_prompt": "dark, blurry images"}
        >>> executor = {"inference_steps": 50, "num_of_exec": 3, "cfg_scale": 7.5}
        >>> images = generate_image(job, pipeline, executor, upscaling_factor=4, parallel_exec=True)
        >>> len(images)  # Should be equal to the number specified in 'num_of_exec'
    """
    prompt = job.get("prompt")
    negative_prompt = job.get("negative_prompt")
    num_inference_steps = executor.get("inference_steps")
    num_images = executor.get("num_of_exec")
    cfg = executor.get("cfg_scale")
    
    def execute_pipeline(num_images):
        return pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images,
            guidance_scale=cfg,
            positive_embeddings=getattr(pipeline, 'positive_embeddings', None),
            negative_embeddings=getattr(pipeline, 'negative_embeddings', None),
            **kwargs
        )

    def upscale_image(image, factor):
        model = RealESRGAN(device, scale=factor)
        model.load_weights('weights/RealESRGAN_x4.pth', download=True)
        
        pil_image = Image.fromarray(image)
        upscaled_image = model.predict(pil_image)
        
        return np.array(upscaled_image)

    try:
        with torch.no_grad():
            if parallel_exec:
                output = execute_pipeline(num_images)
                base_images = output["images"]
            else:
                output = execute_pipeline(1)
                base_images = output["images"]
            
            high_res_images = [upscale_image(img, upscaling_factor) for img in base_images]
            
            return high_res_images
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        return None