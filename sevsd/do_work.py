from sevsd.setup_pipeline import setup_pipeline
from sevsd.process_task import process_task

models = [
    {
        "name": "./model_cache/model1.safetensors",
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

def do_work(models, jobs, image_path, parallel_exec=True, **kwargs):
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
