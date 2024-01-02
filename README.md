# Easy Stable Diffusion Image Generation

![Build Status](https://github.com/leopedroso45/Stable-Diffusion-ImageGen/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/leopedroso45/Stable-Diffusion-ImageGen/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/leopedroso45/Stable-Diffusion-ImageGen)
![License](https://img.shields.io/github/license/leopedroso45/Stable-Diffusion-ImageGen)

`sevsd` is a Python package specifically designed to make the process of generating images using Stable Diffusion models as simple as possible. The package enables image generation with just a single function call, greatly simplifying the integration of Stable Diffusion into various applications. Utilizing Hugging Face's `diffusers` library, `sevsd` provides an intuitive and flexible interface for generating images based on textual prompts. This makes it an ideal choice for building HTTP APIs, high-level services, or any application requiring AI-driven image generation.

## Features

- Simplified interface for Stable Diffusion image generation, enabling the creation of images with just a single function call.
- Easy integration of Stable Diffusion model into Python applications.
- Customizable image generation based on user-defined tasks and configurations.
- Batch processing capabilities for handling multiple tasks efficiently.
- Compatibility with CUDA-enabled GPUs and MPS (Apple's Metal Performance Shaders) for enhanced performance.

## Requirements

- Python 3.11+
- PyTorch
- Hugging Face `diffusers` library
- CUDA-compatible GPU (recommended for better performance)

## Installation

### Install from PyPI

You can install `sevsd` directly from PyPI. This is the recommended way to install the package as it will always provide you with the latest stable version:

```bash
pip install sevsd
```

### Install from Source

If you prefer to install `sevsd` from the source, for example, to get the latest changes that may not be released on PyPI yet, you can clone the repository and install it manually:

```bash
git clone https://github.com/leopedroso45/Stable-Diffusion-ImageGen
cd Stable-Diffusion-ImageGen
pip install .
```

Note: When installing from source, make sure you have the necessary build tools and dependencies installed on your system.

## Usage

Import and use `sevsd` in your Python project:

```python
from sevsd import do_work

# Define your models and jobs
models = [
    {
        "name": './model_cache/model1.safetensors',
        "executor": {
            "labels": [1],
            "num_of_exec": 1,
            "cfg_scale": 7,
            "inference_steps": 100,
        }
    },
    {
        "name": './model_cache/model2.safetensors',
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
        "prompt": 'A scenic landscape',
        "negative_prompt": "blurred image, black and white, watermarked image",
    },
    {
        "label": 2,
        "prompt": 'A person wearing a mask',
        "negative_prompt": 'deformed anatomy, hand-drawn image, blurred image',
    },
]

do_work(models, jobs, './generated-images')
```

This example demonstrates a basic usage scenario. Customize the `models` and `jobs` as needed for your application.

## Components

- `setup_pipeline`: Prepares the Stable Diffusion pipeline with the specified model configuration.
- `process_task`: Processes individual tasks, generating and saving images based on job specifications.
- `generate_image`: Handles the image generation process for each job.
- `setup_device`: Sets up the computation device (GPU or CPU) for image generation.
- `check_os_path`: Ensures the output path exists or creates it.
- `check_cuda_and_clear_cache`: Manages GPU memory and cache for efficient processing.
- `do_work`: Central function to orchestrate the processing of jobs with corresponding models.

## Customization

You can customize the image generation process by adjusting the `models` and `jobs` lists. Define different prompts, model paths, execution parameters, and more to cater to diverse image styles and requirements.

## Note

- Ensure sufficient GPU memory if using CUDA.
- The package is optimized for flexible handling of various job and model configurations.
- For detailed examples and advanced usage, refer to the source code documentation.

## Contributing

Contributions to `sevsd` are welcome! Please refer to the repository's issues and pull requests for ongoing development.

## License

`sevsd` is licensed under the [MIT License](LICENSE).
