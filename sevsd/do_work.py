import argparse
from sevsd.setup_pipeline import setup_pipeline
from sevsd.process_task import process_task

def do_work(configs, tasks, path):
    for config in configs:
        pipeline = setup_pipeline(config)
        process_task(tasks, pipeline, path)