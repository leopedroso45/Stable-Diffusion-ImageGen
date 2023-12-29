# Easy Stable Diffusion Image Generation

`sevsd` is a Python package designed to simplify the integration of Stable Diffusion image generation into various applications. Utilizing Hugging Face's `diffusers` library, `sevsd` provides a straightforward and flexible interface for generating images based on textual prompts. Whether you're building HTTP APIs, high-level services, or other applications, `sevsd` streamlines the process of incorporating AI-driven image generation.

## Features

- Easy integration of Stable Diffusion model into Python applications.
- Customizable image generation based on user-defined tasks and configurations.
- Batch processing capabilities for handling multiple tasks efficiently.
- Compatibility with CUDA-enabled GPUs for enhanced performance.

## Requirements

- Python 3.11+
- PyTorch
- Hugging Face `diffusers` library
- CUDA-compatible GPU (recommended for better performance)

## Installation

Install `sevsd` directly from the source:

```bash
git clone https://github.com/leopedroso45/Stable-Diffusion-ImageGen
cd Stable-Diffusion-ImageGen
pip install .
````

## Usage

Import and use `sevsd` in your Python project:

```python
from sevsd import do_work

# Define your configuration and tasks
configs = [("CompVis/stable-diffusion-v1-4", "./model_cache")]
tasks = [("A scenic landscape", None, 50, 1, 7.5)]

# Process tasks
do_work(configs, tasks, "./generated-images")
```

This example demonstrates a basic usage scenario. Customize the `configs` and `tasks` as needed for your application.

## Components

- `setup_pipeline`: Initializes the Stable Diffusion pipeline with given configurations.
- `process_task`: Processes the list of tasks, generating and saving images.
- `generate_image`: Handles the image generation process for each task.
- `setup_device`: Sets up the computation device (GPU or CPU) for image generation.

## Customization

You can customize the image generation process by modifying the `tasks` list with different prompts, inference steps, and other parameters. The `configs` list allows for different model configurations, enabling diverse image styles.

## Note

- Ensure sufficient GPU memory if using CUDA.
- The package is optimized for batch processing. Modify `tasks` and `configs` to fit your requirements.
- For detailed examples and advanced usage, refer to the source code documentation.

## Contributing

Contributions to `sevsd` are welcome! Please refer to the repository's issues and pull requests for ongoing development.

## License

`sevsd` is licensed under the [MIT License](LICENSE).
