from setuptools import setup, find_packages

setup(
    name="sevsd",
    version="0.3.0",
    author="Leonardo Severo",
    author_email="leopedroso45@gmail.com",
    description="A Python package to make Stable Diffusion Image Generation ridiculously easy",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/leopedroso45/Stable-Diffusion-ImageGen",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    install_requires=[
        "torch",
        "diffusers",
        "transformers",
        "omegaconf",
    ],
    include_package_data=True,
    keywords='image generation, stable diffusion, AI',
    entry_points={},
)
