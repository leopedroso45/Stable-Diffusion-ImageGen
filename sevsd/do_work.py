from sevsd.setup_pipeline import setup_pipeline
from sevsd.process_task import process_task

def do_work(models, jobs, image_path, parallel_exec=True, **kwargs):
    r"""
    Orchestrates the processing of image generation tasks based on given models and jobs.

    This function iterates over each model and the associated jobs, generating images as specified. It sets up the pipeline for each model and executes the image generation tasks, saving the results to the specified path.

    Parameters:
        models (list of dicts): List of model configurations. Each configuration includes:
                                - 'name' (str): The model name or path.
                                - 'executor' (dict): Parameters like 'labels', 'num_of_exec', 'cfg_scale', and 'inference_steps'.
        jobs (list of dicts): List of job configurations. Each job includes:
                              - 'label' (int): Corresponding model label.
                              - 'prompt' (str): Text prompt for image generation.
                              - 'negative_prompt' (str, optional): Text prompt for undesired image features.
        image_path (str): Directory path to save the generated images.
        parallel_exec (bool, optional): Flag to enable parallel execution. Defaults to True.
        **kwargs: Additional keyword arguments for pipeline setup.

    Example:
        models = [
            {
                "name": "CompVis/stable-diffusion-v1-4",
                "executor": {
                    "labels": [1],
                    "num_of_exec": 1,
                    "cfg_scale": 7,
                    "inference_steps": 100,
                }
            },
            {
                "name": "./model_cache/model2.safetensors",
                "executor": {
                    "labels": [2],
                    "num_of_exec": 2,
                    "cfg_scale": 6,
                    "inference_steps": 50,
                }
            },
        ]

        jobs = [
            {
                "label": 1,
                "prompt": "A scenic landscape",
                "negative_prompt": "blurred image, black and white, watermarked image",
            },
            {
                "label": 2,
                "prompt": "A person wearing a mask",
                "negative_prompt": "deformed anatomy, hand-drawn image, blurred image",
            },
        ]

        do_work(models, jobs, "./generated-images")
    """
    job_dict = {job['label']: [] for job in jobs}
    for job in jobs:
        job_dict[job['label']].append(job)

    for model in models:
        pipeline = setup_pipeline(model["name"], **kwargs)
        labels = model.get("executor", {}).get("labels", [])
        for label in labels:
            if label in job_dict:
                for job in job_dict[label]:
                    executor = model.get("executor", {})
                    process_task(job, pipeline, executor, image_path, parallel_exec)
