import torch

def generate_image(args, pipeline):
    prompt, negative_prompt, num_inference_steps, num_images, cfg = args
    try:
        with torch.no_grad():
            output = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        num_images_per_prompt=num_images,
                        guidance_scale=cfg
                    )
            print(f"Available keys in output: {output.keys()}\nNumber of items inside the output is: {len(output['images'])}")
            return output["images"]
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()