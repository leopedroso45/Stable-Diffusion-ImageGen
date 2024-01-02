

from datetime import datetime
import gc
import os
from sevsd.generate_image import generate_image
import torch

def process_task(job, pipeline, executor, path, parallel_exec=True):
    
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        gc.collect()

def check_os_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[sevsd] - created path: {path}")
    return path