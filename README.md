# Stable Diffusion Image Generation Pipeline

This repository contains a Python script for generating images using the Stable Diffusion model from Hugging Face's `diffusers` library. The script is designed to work with a list of tasks and configurations, processing each task with the specified model configuration.

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face `diffusers` library
- CUDA-compatible GPU (optional but recommended for performance)

## Setup

1. Clone the repository.
2. Install required libraries:
   ```bash
   pip install torch diffusers
   ```
3. Prepare your configuration and tasks files (see below for format).

## Configuration

The script expects a configuration file (`pipeline_config/config.py`) with model configurations. Each configuration is a tuple containing the model link or path and cache directory.

Example:
```python
# pipeline_config/config.py

config_1 = [
    ("CompVis/stable-diffusion-v1-4", "./model_cache"),
    # Add more configurations as needed
]
```

## Tasks

Tasks are defined in a separate Python file (`tasks/tasks.py`). Each task is a tuple containing the prompt, negative prompt, number of inference steps, number of images, and CFG scale.

Example:
```python
# tasks/tasks.py

tasks = [
    ("A scenic landscape", None, 50, 1, 7.5),
    # Add more tasks as needed
]
```

## Usage

Run the script with the following command:
```bash
python sd15.py
```

The script will process each task with each configuration in the `configs` list. Generated images will be saved in the `generated-images` directory with a timestamp.

## Image Generation

The `generate_image` function handles the image generation process. It takes a task and a pipeline object, generates the image(s) based on the provided prompt, and returns the images.

## Processing Tasks

The `process_task` function iterates over all tasks, generates images, and saves them to disk. If image generation fails (typically due to memory constraints), it prints an error message.

## Main Execution

The `do_work` function iterates over all configurations, sets up the pipeline, and processes the tasks.

## Note

- Ensure you have enough GPU memory if running on a CUDA device.
- The script is designed for batch processing of multiple tasks. Modify the `tasks` and `configs` as per your requirements.

## License

[MIT](LICENSE)
