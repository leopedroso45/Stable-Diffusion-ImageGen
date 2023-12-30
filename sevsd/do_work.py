from sevsd.setup_pipeline import setup_pipeline
from sevsd.process_task import process_task

def do_work(configs, tasks, path, parallel_exec=True, **kwargs):
    task_dict = {task["task_id"]: [] for task in tasks}
    for task in tasks:
        task_dict[task["task_id"]].append(task)

    for config in configs:
        pipeline = setup_pipeline(config["model_info"], **kwargs)
        task_ids = config.get("task_ids", [])
        for task_id in task_ids:
            if task_id in task_dict:
                for task in task_dict[task_id]:
                    process_task(task["details"], pipeline, path, parallel_exec)
