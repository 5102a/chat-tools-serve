import json
import os
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UniPCMultistepScheduler
import torch
import torch


model_path = "models/stable-diffusion-v1-5"


def getPath(path=""):
    p = os.path.abspath(os.path.join(model_path, path))
    print(f"path:{p}")
    return p


pipe = StableDiffusionPipeline.from_single_file(
    getPath("v1-5-pruned-emaonly.safetensors"), load_safety_checker=True, extract_ema=True, use_safetensors=True, torch_dtype=torch.float16, device_map="auto")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# pipe.load_lora_weights(getPath("lora/school_gym_v0.1.safetensors"))

pipe.load_textual_inversion(
    getPath("embeddings/easynegative.safetensors"), token="easynegative")

pipe = pipe.to("cuda")

class GenerateImage:
    def gen(prompt: str = ""):
        print(prompt)
        if prompt == "":
            return ""

        with autocast("cuda"):
            image = pipe(prompt="8k,masterpiece,best quality,[perfect face],official art,ultra highres," + prompt,
                         negative_prompt="easynegative,worst quality,low quality,normal quality,lowres,watermark,monochrome,grayscale,ugly,blurry,Tan skin, dark skin, black skin, skin spots, skin blemishes, age spot, glans, disabled, distorted, bad anatomy, morbid, malformation, amputation, bad proportions, twins, missing body, fused body, extra head, poorly drawn face, bad eyes, deformed eye, unclear eyes, cross-eyed, long neck, malformed limbs, extra limbs, extra arms, missing arms, bad tongue, strange fingers, mutated hands, missing hands, poorly drawn hands, extra hands, fused hands, connected hand, bad hands,, deformed hands, extra legs, bad legs, many legs, more than two legs, bad feet, wrong feet, extra feets",
                         num_inference_steps=60, height=512, width=512).images[0]

        image.save("astronaut_rides_horse.png")
        return "./astronaut_rides_horse.png"
