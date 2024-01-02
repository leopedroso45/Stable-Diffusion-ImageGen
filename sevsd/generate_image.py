import torch
        
def generate_image(job, pipeline, executor, parallel_exec=True, **kwargs):

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
            **kwargs
        )
    try:
        with torch.no_grad():
            if parallel_exec:
                output = execute_pipeline(num_images)
                return output["images"]
            else:
                output = execute_pipeline(1)
                return output["images"]
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        return None