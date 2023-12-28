# Image Generation Pipeline Configuration

This folder contains the configuration file for the image generation pipeline using the Stable Diffusion model.

## How to Configure

The configuration file (`config.py`) should contain the model configurations. Each configuration is a tuple containing the link or path to the pre-trained model and the cache directory.

### Configuration File Structure

The `config.py` file should be structured as follows:

```python
config_1 = [
    ("model_link_or_path", "cache_directory"),
    # Add more configurations as needed
]
```

Example:

```python
config_1 = [
    ("CompVis/stable-diffusion-v1-5", "./model_cache"),
    ("CompVis/stable-diffusion-v2-0", "./model_cache")
]
```