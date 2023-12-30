import torch

def check_and_clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def generate_image(task, pipeline, parallel_exec=True, **kwargs):
    prompt, negative_prompt, num_inference_steps, num_images, cfg = task["details"]
    
    def execute_pipeline(num_images):
        check_and_clear_cache()
        return pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images,
            guidance_scale=cfg,
            **kwargs
        )
    
    try:
        with torch.no_grad():
            if parallel_exec:
                output = execute_pipeline(num_images)
                return output["images"]
            else:
                output_list = [execute_pipeline(1)["images"][0] for _ in range(num_images)]
                return output_list
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        return None
    finally:
        check_and_clear_cache()