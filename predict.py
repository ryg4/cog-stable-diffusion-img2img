import os
from typing import List

import cv2
import numpy as np

import torch
from torch import Tensor
import moviepy
from moviepy.editor import *

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline, 
    ControlNetModel,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from PIL import Image
from cog import BasePredictor, Input, Path

MODEL_ID = "stabilityai/stable-diffusion-1-5"
MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=False,
        ).to("cuda")
        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        self.img2img_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=self.controlnet, 
            torch_dtype=torch.float16).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="A fantasy landscape, trending on artstation",
        ),
        negative_prompt: str = Input(
            description="The prompt NOT to guide the image generation. Ignored when not using guidance",
            default=None,
        ),
        image: Path = Input(
            description="Inital image to generate variations of.",
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when providing the image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output. Higher number of outputs may OOM.",
            ge=1,
            le=8,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=["DDIM", "K_EULER", "DPMSolverMultistep", "K_EULER_ANCESTRAL", "PNDM", "KLMS"],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        cannyinput = np.array(image)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        pipe = self.img2img_pipe
        extra_kwargs = {
            "image": Image.open(image).convert("RGB"),
            "strength": prompt_strength,
        }
        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)
        prompts = prompt.split("|")
        prompt_embedding_list = []
        print(prompts)
        for prompt in prompts:
            print(f"getting embeddings for {prompt}")
            text_inputs = self.txt2img_pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.txt2img_pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = self.txt2img_pipe.text_encoder(
                text_input_ids.to("cuda"),
            )[0]
            prompt_embedding_list.append(prompt_embeds)

        tweened_prompt_embeds = []
        for i in range(0, len(prompt_embedding_list) - 1):
            for j in np.arange(0, 1, 0.1):
                print(f"tweening between {prompt_embedding_list[i]} and {prompt_embedding_list[i + 1]} with value {j}")
                tweened_prompt_embeds.append(weighted_sum(prompt_embedding_list[i], prompt_embedding_list[i + 1], j))

        output_paths = []
        output_path_strings = []
        i = 0
        for embeds in tweened_prompt_embeds:
            print(f"running {i} of {len(tweened_prompt_embeds)}")
            generator = torch.Generator("cuda").manual_seed(seed)
            output = pipe(
                prompt=None,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                prompt_embeds=embeds,
                **extra_kwargs,
            )

            for j, sample in enumerate(output.images):
                output_path = f"/tmp/out-{i}.png"
                i += 1
                sample.save(output_path)
                output_path_strings.append(output_path)
                output_paths.append(Path(output_path))
        
        clips = [ImageClip(m).set_duration(0.1) for m in output_path_strings]

        concat_clip = concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(f"/tmp/test.mp4", fps=10)

        return [Path("/tmp/test.mp4")]

def weighted_sum(condA:Tensor, condB:Tensor, alpha:float) -> Tensor:
    ''' linear interpolate on latent space of condition '''

    return (1 - alpha) * condA + (alpha) * condB

def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
