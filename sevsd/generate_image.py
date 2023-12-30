import torch

def generate_image(args, pipeline, parallel_exec, **kwargs):
    prompt, negative_prompt, num_inference_steps, num_images, cfg = args
    
    def execute_pipeline(num_images):
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()