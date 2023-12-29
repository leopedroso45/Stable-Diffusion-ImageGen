

from datetime import datetime
import gc
import os
from sevsd.generate_image import generate_image
import torch

def process_task(tasks, pipeline, path):
    try:
        path = check_os_path(path)
        if tasks is not None:
            for task in tasks:
                images = generate_image(task, pipeline)
                if images is not None:
                    for image in images:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                        image_path = f"{path}/generated_image_{timestamp}.png"
                        image.save(image_path)
                        print(f"Image saved at {image_path}")
                else:
                    print("Image generation failed due to memory constraints.")
                check_cuda_and_clear_cache()
    except Exception as e:
        print(f"Exception: {e}")
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
        print(f"Created path: {path}")
    return path