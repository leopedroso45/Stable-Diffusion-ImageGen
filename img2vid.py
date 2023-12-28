import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)

pipe.to("cuda")
pipe.unet.enable_forward_chunking()

image = load_image("./generated-images2/0.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator, num_frames=25).frames[0]

export_to_video(frames, "generated.mp4", fps=7)