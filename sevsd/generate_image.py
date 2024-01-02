import torch
        
def generate_image(job, pipeline, executor, parallel_exec=True, **kwargs):
    r"""
    Generates images based on textual prompts using the provided Stable Diffusion pipeline.

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
        parallel_exec (bool, optional): If True, generates 'num_of_exec' images in parallel. Otherwise, generates images sequentially. Defaults to True.
        **kwargs: Additional keyword arguments passed to the pipeline function.

    Returns:
        list or None: A list of generated images if successful, or None if a RuntimeError occurs during image generation.

    Raises:
        RuntimeError: If an error occurs during the image generation process.

    Example:
        >>> job = {"prompt": "A beautiful landscape", "negative_prompt": "dark, blurry images"}
        >>> executor = {"inference_steps": 50, "num_of_exec": 3, "cfg_scale": 7.5}
        >>> images = generate_image(job, pipeline, executor, parallel_exec=True)
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