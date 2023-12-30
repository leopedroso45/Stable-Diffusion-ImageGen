from sevsd.setup_pipeline import setup_pipeline
from sevsd.process_task import process_task

def do_work(configs, tasks, path, parallel_exec=True, **kwargs):
    for config in configs:
        pipeline = setup_pipeline(config, **kwargs)
        process_task(tasks, pipeline, path, parallel_exec)