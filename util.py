import requests
import json

import os
import random

import gradio as gr
import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, DiffusionPipeline
import uuid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "/data/SSD-1B",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to(device)
refiner.to(device)
print("Loaded on Device!")

def gen_llm_resp(input: str):
    url = "http://gpu:8000/v1/chat/completions"
    payload = json.dumps({
        "model": "qwen-q",
        "messages": [
            {
                "role": "user",
                "content": f"{input}"
            }
        ],
        "max_length": 250,
        "temperature": 0.9
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return json.loads(response.text)['choices'][0]['message']['content']


MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def save_image(img):
    unique_name = str(uuid.uuid4()) + '.png'
    img.save(unique_name)
    return unique_name


DEFAULT_STYLE_NAME = "Cinematic"


def gen_image(prompt: str,
              negative_prompt: str = "",
              style: str = DEFAULT_STYLE_NAME,
              prompt_2: str = "",
              negative_prompt_2: str = "",
              use_negative_prompt: bool = False,
              use_prompt_2: bool = False,
              use_negative_prompt_2: bool = False,
              seed: int = 0,
              width: int = 1024,
              height: int = 1024,
              guidance_scale_base: float = 5.0,
              guidance_scale_refiner: float = 5.0,
              num_inference_steps_base: int = 25,
              num_inference_steps_refiner: int = 25,
              apply_refiner: bool = False,
              randomize_seed: bool = False, ):
    seed = randomize_seed_fn(seed, randomize_seed)
    generator = torch.Generator().manual_seed(seed)
    latents = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_2=prompt_2,
        negative_prompt_2=negative_prompt_2,
        width=width,
        height=height,
        guidance_scale=guidance_scale_base,
        num_inference_steps=num_inference_steps_base,
        generator=generator,
        output_type="latent",
    ).images[0]
    image = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_2=prompt_2,
        negative_prompt_2=negative_prompt_2,
        guidance_scale=guidance_scale_refiner,
        num_inference_steps=num_inference_steps_refiner,
        image=latents,
        generator=generator,
    ).images[0]
    image_path = save_image(image)
    print(image_path)
    return image_path


def gen_image_with_controlnet():
    return ""
