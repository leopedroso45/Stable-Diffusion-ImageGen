

from datetime import datetime
import gc
import os
from sevsd.generate_image import generate_image
import torch

def process_task(job, pipeline, executor, path, parallel_exec=True):
    r"""
    Processes a single image generation job using the specified pipeline and execution parameters.

    This function handles the generation of one or more images based on a given job description. It supports both parallel and sequential execution modes. Generated images are saved to the specified path.

    Parameters:
        job (dict): A dictionary containing details for the image generation task. It includes 'prompt' and optionally 'negative_prompt'.
        pipeline (callable): The Stable Diffusion pipeline callable used for generating images.
        executor (dict): A dictionary containing execution parameters such as 'num_of_exec', 'cfg_scale', and 'inference_steps'.
        path (str): The directory path where generated images will be saved.
        parallel_exec (bool, optional): If True, generates all specified images in parallel. Defaults to True.

    The function saves each generated image with a unique timestamp in the specified path and prints the save location. In case of any exceptions, they are caught and printed.

    Example:
        job = {
            "prompt": "A scenic landscape",
            "negative_prompt": "blurred image, black and white, watermarked image"
        }
        executor = {
            "num_of_exec": 2,
            "cfg_scale": 7,
            "inference_steps": 50
        }
        pipeline = setup_pipeline("CompVis/stable-diffusion-v1-4")
        process_task(job, pipeline, executor, "./generated-images", parallel_exec=False)

    Note:
        This function also handles CUDA cache clearing and garbage collection for memory management.
    """
    
    def call_generate_image():
        images = generate_image(job, pipeline, executor, parallel_exec)
        if images is not None:
            for image in images:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                image_path = f"{path}/generated_image_{timestamp}.png"
                image.save(image_path)
                print(f"[sevsd] - image saved at {image_path}")
        else:
            print("[sevsd] - image generation failed due to memory constraints.")
        check_cuda_and_clear_cache()
    
    try:
        path = check_os_path(path)
        if job is not None:
            if parallel_exec is not True:
                num_images = executor.get("num_of_exec", 1)
                for _ in range(num_images):
                    call_generate_image()
            else:
                call_generate_image()
    except Exception as e:
        print(f"[sevsd] - exception: {e}")
    finally:
        check_cuda_and_clear_cache()

def check_cuda_and_clear_cache():
    r"""
    Clears the CUDA cache if available, otherwise performs garbage collection.
    This function is called to manage memory usage, particularly when working with large models or multiple image generations.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        gc.collect()

def check_os_path(path):
    r"""
    Checks if the given path exists, and if not, creates the necessary directories.
    This function ensures that the output path for saving images is available.

    Parameters:
        path (str): The directory path to check and create if necessary.

    Returns:
        str: The verified or created directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[sevsd] - created path: {path}")
    return path