# Task Configuration for Image Generation

This folder contains the file to define tasks for image generation using the Stable Diffusion model.

## How to Define Tasks

Tasks are defined in the `tasks.py` file. Each task is a tuple containing the image prompt, negative prompt, number of inference steps, number of images, and CFG scale.

### Task File Structure

The `tasks.py` file should be structured as follows:

```python
tasks = [
    ("image_prompt", "negative_prompt", number_of_steps, number_of_images, cfg_scale),
    # Add more tasks as needed
]
```

Example:

```python
tasks = [
    ("A scenic landscape", "low res", 50, 1, 7.5),
]
```