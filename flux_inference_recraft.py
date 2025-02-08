import argparse
import copy
import math
import random
from typing import Any
import pdb
import os

import time
from PIL import Image, ImageOps

import torch
from accelerate import Accelerator
from library.device_utils import clean_memory_on_device
from safetensors.torch import load_file
from networks import lora_flux

from library import flux_models, flux_train_utils_recraft as flux_train_utils, flux_utils, sd3_train_utils, \
    strategy_base, strategy_flux, train_util
from torchvision import transforms
import train_network
from library.utils import setup_logging
from diffusers.utils import load_image
import numpy as np

setup_logging()
import logging

logger = logging.getLogger(__name__)


def load_target_model(
        fp8_base: bool,
        pretrained_model_name_or_path: str,
        disable_mmap_load_safetensors: bool,
        clip_l_path: str,
        fp8_base_unet: bool,
        t5xxl_path: str,
        ae_path: str,
        weight_dtype: torch.dtype,
        accelerator: Accelerator
):
    # Determine the loading data type
    loading_dtype = None if fp8_base else weight_dtype

    # Load the main model to the accelerator's device
    _, model = flux_utils.load_flow_model(
        pretrained_model_name_or_path,
        # loading_dtype,
        torch.float8_e4m3fn,
        # accelerator.device,  # Changed from "cpu" to accelerator.device
        "cpu",
        disable_mmap=disable_mmap_load_safetensors
    )

    if fp8_base:
        # Check dtype of the model
        if model.dtype in {torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz}:
            raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
        elif model.dtype == torch.float8_e4m3fn:
            logger.info("Loaded fp8 FLUX model")

    # Load the CLIP model to the accelerator's device
    clip_l = flux_utils.load_clip_l(
        clip_l_path,
        weight_dtype,
        # accelerator.device,  # Changed from "cpu" to accelerator.device
        "cpu",
        disable_mmap=disable_mmap_load_safetensors
    )
    clip_l.eval()

    # Determine the loading data type for T5XXL
    if fp8_base and not fp8_base_unet:
        loading_dtype_t5xxl = None  # as is
    else:
        loading_dtype_t5xxl = weight_dtype

    # Load the T5XXL model to the accelerator's device
    t5xxl = flux_utils.load_t5xxl(
        t5xxl_path,
        loading_dtype_t5xxl,
        # accelerator.device,  # Changed from "cpu" to accelerator.device
        "cpu",
        disable_mmap=disable_mmap_load_safetensors
    )
    t5xxl.eval()

    if fp8_base and not fp8_base_unet:
        # Check dtype of the T5XXL model
        if t5xxl.dtype in {torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz}:
            raise ValueError(f"Unsupported fp8 model dtype: {t5xxl.dtype}")
        elif t5xxl.dtype == torch.float8_e4m3fn:
            logger.info("Loaded fp8 T5XXL model")

    # Load the AE model to the accelerator's device
    ae = flux_utils.load_ae(
        ae_path,
        weight_dtype,
        # accelerator.device,  # Changed from "cpu" to accelerator.device
        "cpu",
        disable_mmap=disable_mmap_load_safetensors
    )

    # # Wrap models with Accelerator for potential distributed setups
    # model, clip_l, t5xxl, ae = accelerator.prepare(model, clip_l, t5xxl, ae)

    return flux_utils.MODEL_VERSION_FLUX_V1, [clip_l, t5xxl], ae, model


import torchvision.transforms as transforms


class ResizeWithPadding:
    def __init__(self, size, fill=255):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL Image or a NumPy array")

        width, height = img.size

        if width == height:
            img = img.resize((self.size, self.size), Image.LANCZOS)
        else:
            max_dim = max(width, height)

            new_img = Image.new("RGB", (max_dim, max_dim), (self.fill, self.fill, self.fill))
            new_img.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))

            img = new_img.resize((self.size, self.size), Image.LANCZOS)

        return img


def sample(args, accelerator, vae, text_encoder, flux, output_dir, sample_images, sample_prompts):
    def encode_images_to_latents(vae, images):
        # Get image dimensions
        b, c, h, w = images.shape
        num_split = 2 if args.frame_num == 4 else 3
        # Split the image into three parts
        img_parts = [images[:, :, :, i * w // num_split:(i + 1) * w // num_split] for i in range(num_split)]
        # Encode each part
        latents = [vae.encode(img) for img in img_parts]
        # Concatenate latents in the latent space to reconstruct the full image
        latents = torch.cat(latents, dim=-1)
        return latents

    def encode_images_to_latents2(vae, images):
        latents = vae.encode(images)
        return latents

    # Directly use precomputed conditions
    conditions = {}
    with torch.no_grad():
        for image_path, prompt_dict in zip(sample_images, sample_prompts):
            prompt = prompt_dict.get("prompt", "")
            if prompt not in conditions:
                logger.info(f"Cache conditions for image: {image_path} with prompt: {prompt}")
                resize_transform = ResizeWithPadding(size=512, fill=255) if args.frame_num == 4 else ResizeWithPadding(size=352, fill=255)
                img_transforms = transforms.Compose([
                    resize_transform,
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                # Load and preprocess image
                image = img_transforms(np.array(load_image(image_path), dtype=np.uint8)).unsqueeze(0).to(
                    # accelerator.device,  # Move image to CUDA
                    vae.device,
                    dtype=vae.dtype
                )
                latents = encode_images_to_latents2(vae, image)

                # Log the shape of latents
                logger.debug(f"Encoded latents shape for prompt '{prompt}': {latents.shape}")
                # Store conditions on CUDA
                # conditions[prompt] = latents[:,:,latents.shape[2]//2:latents.shape[2], :latents.shape[3]//2].to("cpu")
                conditions[prompt] = latents.to("cpu")

    sample_conditions = conditions

    if sample_conditions is not None:
        conditions = {k: v for k, v in sample_conditions.items()}  # Already on CUDA

    sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
    text_encoder[0].to(accelerator.device)
    text_encoder[1].to(accelerator.device)

    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(512)
    text_encoding_strategy = strategy_flux.FluxTextEncodingStrategy(True)

    with accelerator.autocast(), torch.no_grad():
        for prompt_dict in sample_prompts:
            for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                if p not in sample_prompts_te_outputs:
                    logger.info(f"Cache Text Encoder outputs for prompt: {p}")
                    tokens_and_masks = tokenize_strategy.tokenize(p)
                    sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                        tokenize_strategy, text_encoder, tokens_and_masks, True
                    )

    logger.info(f"Generating image")
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad(), accelerator.autocast():
        for prompt_dict in sample_prompts:
            sample_image_inference(
                args,
                accelerator,
                flux,
                text_encoder,
                vae,
                save_dir,
                prompt_dict,
                sample_prompts_te_outputs,
                None,
                conditions
            )

    clean_memory_on_device(accelerator.device)


def sample_image_inference(
        args,
        accelerator: Accelerator,
        flux: flux_models.Flux,
        text_encoder,
        ae: flux_models.AutoEncoder,
        save_dir,
        prompt_dict,
        sample_prompts_te_outputs,
        prompt_replacement,
        sample_images_ae_outputs
):
    # Extract parameters from prompt_dict
    sample_steps = prompt_dict.get("sample_steps", 20)
    width = prompt_dict.get("width", 1024) if args.frame_num == 4 else prompt_dict.get("width", 1056)
    height = prompt_dict.get("height", 1024) if args.frame_num == 4 else prompt_dict.get("height", 1056)
    scale = prompt_dict.get("scale", 1.0)
    seed = prompt_dict.get("seed")
    prompt: str = prompt_dict.get("prompt", "")

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        torch.cuda.seed()

    # Ensure height and width are divisible by 16
    height = max(64, height - height % 16)
    width = max(64, width - width % 16)
    logger.info(f"prompt: {prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {scale}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    # Encode prompts
    # Assuming that TokenizeStrategy and TextEncodingStrategy are compatible with Accelerator
    text_encoder_conds = []
    if sample_prompts_te_outputs and prompt in sample_prompts_te_outputs:
        text_encoder_conds = sample_prompts_te_outputs[prompt]
        logger.info(f"Using cached text encoder outputs for prompt: {prompt}")

    if sample_images_ae_outputs and prompt in sample_images_ae_outputs:
        ae_outputs = sample_images_ae_outputs[prompt]
    else:
        ae_outputs = None

    # ae_outputs = torch.load('ae_outputs.pth', map_location='cuda:0')

    # text_encoder_conds = torch.load('text_encoder_conds.pth', map_location='cuda:0')
    l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds

    # 打印调试信息
    logger.debug(
        f"l_pooled shape: {l_pooled.shape}, t5_out shape: {t5_out.shape}, txt_ids shape: {txt_ids.shape}, t5_attn_mask shape: {t5_attn_mask.shape}")

    # 采样图像
    weight_dtype = ae.dtype  # TODO: give dtype as argument
    packed_latent_height = height // 16
    packed_latent_width = width // 16

    # 打印调试信息
    logger.debug(f"packed_latent_height: {packed_latent_height}, packed_latent_width: {packed_latent_width}")

    # 准备噪声张量在 CUDA 上
    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=accelerator.device,
        dtype=weight_dtype,
        generator=torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None,
    )

    timesteps = flux_train_utils.get_schedule(sample_steps, noise.shape[1], shift=True)  # FLUX.1 dev -> shift=True
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(
        accelerator.device, dtype=weight_dtype
    )
    t5_attn_mask = t5_attn_mask.to(accelerator.device)

    clip_l, t5xxl = text_encoder
    # ae.to("cpu")
    clip_l.to("cpu")
    t5xxl.to("cpu")

    clean_memory_on_device(accelerator.device)
    flux.to("cuda")

    for param in flux.parameters():
        param.requires_grad = False

    # 执行去噪
    with accelerator.autocast(), torch.no_grad():
        x = flux_train_utils.denoise(args, flux, noise, img_ids, t5_out, txt_ids, l_pooled, timesteps=timesteps,
                                     guidance=scale, t5_attn_mask=t5_attn_mask, ae_outputs=ae_outputs)

    # 打印x的形状
    logger.debug(f"x shape after denoise: {x.shape}")

    x = x.float()
    x = flux_utils.unpack_latents(x, packed_latent_height, packed_latent_width)

    # 将潜在向量转换为图像
    # clean_memory_on_device(accelerator.device)
    ae.to(accelerator.device)
    with accelerator.autocast(), torch.no_grad():
        x = ae.decode(x)
    ae.to("cpu")
    clean_memory_on_device(accelerator.device)

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    # 生成唯一的文件名
    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict.get("enum", 0)  # Ensure 'enum' exists
    img_filename = f"{ts_str}{seed_suffix}_{i}.png"  # Added 'i' to filename for uniqueness
    image.save(os.path.join(save_dir, img_filename))


def setup_argparse():
    parser = argparse.ArgumentParser(description="FLUX-Controlnet-Inpainting Inference Script")

    # Paths
    parser.add_argument('--base_flux_checkpoint', type=str, required=True,
                        help='Path to BASE_FLUX_CHECKPOINT')
    parser.add_argument('--lora_weights_path', type=str, required=True,
                        help='Path to LORA_WEIGHTS_PATH')
    parser.add_argument('--clip_l_path', type=str, required=True,
                        help='Path to CLIP_L_PATH')
    parser.add_argument('--t5xxl_path', type=str, required=True,
                        help='Path to T5XXL_PATH')
    parser.add_argument('--ae_path', type=str, required=True,
                        help='Path to AE_PATH')
    parser.add_argument('--sample_images_file', type=str, required=True,
                        help='Path to SAMPLE_IMAGES_FILE')
    parser.add_argument('--sample_prompts_file', type=str, required=True,
                        help='Path to SAMPLE_PROMPTS_FILE')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save OUTPUT_DIR')
    parser.add_argument('--frame_num', type=int, choices=[4, 9], required=True,
                        help="The number of steps in the generated step diagram (choose 4 or 9)")

    return parser.parse_args()


def main(args):
    accelerator = Accelerator(mixed_precision='bf16', device_placement=True)

    BASE_FLUX_CHECKPOINT = args.base_flux_checkpoint
    LORA_WEIGHTS_PATH = args.lora_weights_path
    CLIP_L_PATH = args.clip_l_path
    T5XXL_PATH = args.t5xxl_path
    AE_PATH = args.ae_path

    SAMPLE_IMAGES_FILE = args.sample_images_file
    SAMPLE_PROMPTS_FILE = args.sample_prompts_file
    OUTPUT_DIR = args.output_dir

    with open(SAMPLE_IMAGES_FILE, "r", encoding="utf-8") as f:
        image_lines = f.readlines()
    sample_images = [line.strip() for line in image_lines if line.strip() and not line.strip().startswith("#")]

    sample_prompts = train_util.load_prompts(SAMPLE_PROMPTS_FILE)

    # Load models onto CUDA via Accelerator
    _, [clip_l, t5xxl], ae, model = load_target_model(
        fp8_base=True,
        pretrained_model_name_or_path=BASE_FLUX_CHECKPOINT,
        disable_mmap_load_safetensors=False,
        clip_l_path=CLIP_L_PATH,
        fp8_base_unet=False,
        t5xxl_path=T5XXL_PATH,
        ae_path=AE_PATH,
        weight_dtype=torch.bfloat16,
        accelerator=accelerator
    )

    model.eval()
    clip_l.eval()
    t5xxl.eval()
    ae.eval()

    # LoRA
    multiplier = 1.0
    weights_sd = load_file(LORA_WEIGHTS_PATH)
    lora_model, _ = lora_flux.create_network_from_weights(multiplier, None, ae, [clip_l, t5xxl], model, weights_sd,
                                                          True)

    lora_model.apply_to([clip_l, t5xxl], model)
    info = lora_model.load_state_dict(weights_sd, strict=True)
    logger.info(f"Loaded LoRA weights from {LORA_WEIGHTS_PATH}: {info}")
    lora_model.eval()
    lora_model.to("cuda")

    # Set text encoders
    text_encoder = [clip_l, t5xxl]

    sample(args, accelerator, vae=ae, text_encoder=text_encoder, flux=model, output_dir=OUTPUT_DIR,
           sample_images=sample_images, sample_prompts=sample_prompts)


if __name__ == "__main__":
    args = setup_argparse()

    main(args)
