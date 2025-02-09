import spaces
import gradio as gr
import torch
import numpy as np
from PIL import Image
from accelerate import Accelerator
import os
import time
from torchvision import transforms
from safetensors.torch import load_file
from networks import lora_flux
from library import flux_utils, flux_train_utils_recraft as flux_train_utils, strategy_flux
import logging
from huggingface_hub import login
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

accelerator = Accelerator(mixed_precision='bf16', device_placement=True)

hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# Model paths dynamically retrieved using selected model
model_paths = {
    'Wood Sculpture': {
        'BASE_FLUX_CHECKPOINT': "showlab/makeanything",
        'BASE_FILE': "flux_merge_lora/flux_merge_4f_wood-fp8_e4m3fn.safetensors",
        'LORA_REPO': "showlab/makeanything",
        'LORA_FILE': "recraft/recraft_4f_wood_sculpture.safetensors",
        "Frame": 4
    },
    'LEGO': {
        'BASE_FLUX_CHECKPOINT': "showlab/makeanything",
        'BASE_FILE': "flux_merge_lora/flux_merge_9f_lego-fp8_e4m3fn.safetensors",
        'LORA_REPO': "showlab/makeanything",
        'LORA_FILE': "recraft/recraft_9f_lego.safetensors",
        "Frame": 9
    },
    'Sketch': {
        'BASE_FLUX_CHECKPOINT': "showlab/makeanything",
        'BASE_FILE': "flux_merge_lora/flux_merge_9f_portrait-fp8_e4m3fn.safetensors",
        'LORA_REPO': "showlab/makeanything",
        'LORA_FILE': "recraft/recraft_9f_sketch.safetensors",
        "Frame": 9
    },
    'Portrait': {
        'BASE_FLUX_CHECKPOINT': "showlab/makeanything",
        'BASE_FILE': "flux_merge_lora/flux_merge_9f_sketch-fp8_e4m3fn.safetensors",
        'LORA_REPO': "showlab/makeanything",
        'LORA_FILE': "recraft/recraft_9f_portrait.safetensors",
        "Frame": 9
    }
}

# Common paths
clip_repo_id = "comfyanonymous/flux_text_encoders"
t5xxl_file = "t5xxl_fp8_e4m3fn.safetensors"
clip_l_file = "clip_l.safetensors"
ae_repo_id = "black-forest-labs/FLUX.1-dev"
ae_file = "ae.safetensors"


# Model placeholders
model = None
clip_l = None
t5xxl = None
ae = None
lora_model = None

# Function to load a file from Hugging Face Hub
def download_file(repo_id, file_name):
    return hf_hub_download(repo_id=repo_id, filename=file_name)

# Load model function
def load_target_model(selected_model):
    global model, clip_l, t5xxl, ae, lora_model

    # Fetch paths based on the selected model
    model_path = model_paths[selected_model]
    base_checkpoint_repo = model_path['BASE_FLUX_CHECKPOINT']
    base_checkpoint_file = model_path['BASE_FILE']
    lora_repo = model_path['LORA_REPO']
    lora_file = model_path['LORA_FILE']

    # Download necessary files
    BASE_FLUX_CHECKPOINT = download_file(base_checkpoint_repo, base_checkpoint_file)
    CLIP_L_PATH = download_file(clip_repo_id, clip_l_file)
    T5XXL_PATH = download_file(clip_repo_id, t5xxl_file)
    AE_PATH = download_file(ae_repo_id, ae_file)
    LORA_WEIGHTS_PATH = download_file(lora_repo, lora_file)

    logger.info("Loading models...")
    try:
        if model is None is None or clip_l is None or t5xxl is None or ae is None:
            _, model = flux_utils.load_flow_model(
                BASE_FLUX_CHECKPOINT, torch.float8_e4m3fn, "cpu", disable_mmap=False
            )
            clip_l = flux_utils.load_clip_l(CLIP_L_PATH, torch.bfloat16, "cpu", disable_mmap=False)
            clip_l.eval()
            t5xxl = flux_utils.load_t5xxl(T5XXL_PATH, torch.bfloat16, "cpu", disable_mmap=False)
            t5xxl.eval()
            ae = flux_utils.load_ae(AE_PATH, torch.bfloat16, "cpu", disable_mmap=False)

        # Load LoRA weights
        multiplier = 1.0
        weights_sd = load_file(LORA_WEIGHTS_PATH)
        lora_model, _ = lora_flux.create_network_from_weights(multiplier, None, ae, [clip_l, t5xxl], model, weights_sd, True)
        lora_model.apply_to([clip_l, t5xxl], model)
        info = lora_model.load_state_dict(weights_sd, strict=True)
        logger.info(f"Loaded LoRA weights from {LORA_WEIGHTS_PATH}: {info}")
        lora_model.eval()

        logger.info("Models loaded successfully.")
        return "Models loaded successfully."

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return f"Error loading models: {e}"

# Image pre-processing (resize and padding)
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

        # # Convert to RGB to remove transparency, fill with white background if necessary
        # if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        #     background = Image.new("RGB", img.size, (fill, fill, fill))
        #     background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
        #     img = background

        # if width == height:
        #     img = img.resize((self.size, self.size), Image.LANCZOS)
        # else:
        max_dim = max(width, height)
        new_img = Image.new("RGB", (max_dim, max_dim), (self.fill, self.fill, self.fill))
        new_img.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))
        img = new_img.resize((self.size, self.size), Image.LANCZOS)
        return img

# The function to generate image from a prompt and conditional image
@spaces.GPU(duration=180)
def infer(prompt, sample_image, recraft_model, seed=0):
    global model, clip_l, t5xxl, ae, lora_model
    if model is None or lora_model is None or clip_l is None or t5xxl is None or ae is None:
        logger.error("Models not loaded. Please load the models first.")
        return None

    model_path = model_paths[recraft_model]
    frame_num = model_path['Frame']

    logger.info(f"Started generating image with prompt: {prompt}")

    lora_model.to("cuda")
    
    model.eval()
    clip_l.eval()
    t5xxl.eval()
    ae.eval()

    # # Load models
    # model, [clip_l, t5xxl], ae = load_target_model()

    # # LoRA
    # multiplier = 1.0
    # weights_sd = load_file(LORA_WEIGHTS_PATH)
    # lora_model, _ = lora_flux.create_network_from_weights(multiplier, None, ae, [clip_l, t5xxl], model, weights_sd,
    #                                                       True)

    # lora_model.apply_to([clip_l, t5xxl], model)
    # info = lora_model.load_state_dict(weights_sd, strict=True)
    # logger.info(f"Loaded LoRA weights from {LORA_WEIGHTS_PATH}: {info}")
    # lora_model.eval()
    # lora_model.to(device)

    logger.debug(f"Using seed: {seed}")

    # Preprocess the conditional image
    resize_transform = ResizeWithPadding(size=512) if frame_num == 4 else ResizeWithPadding(size=352)
    img_transforms = transforms.Compose([
        resize_transform,
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    image = img_transforms(np.array(sample_image, dtype=np.uint8)).unsqueeze(0).to(
        device=device,
        dtype=torch.bfloat16
    )
    logger.debug("Conditional image preprocessed.")

    # Encode the image to latents
    ae.to(device)
    latents = ae.encode(image)
    logger.debug("Image encoded to latents.")

    conditions = {}
    # conditions[prompt] = latents.to("cpu")
    conditions[prompt] = latents


    # ae.to("cpu")
    clip_l.to(device)
    t5xxl.to(device)

    # Encode the prompt
    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(512)
    text_encoding_strategy = strategy_flux.FluxTextEncodingStrategy(True)
    tokens_and_masks = tokenize_strategy.tokenize(prompt)
    l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, t5xxl], tokens_and_masks, True)

    logger.debug("Prompt encoded.")

    # Prepare the noise and other parameters
    width = 1024 if frame_num == 4 else 1056
    height = 1024 if frame_num == 4 else 1056

    height = max(64, height - height % 16)
    width = max(64, width - width % 16)

    packed_latent_height = height // 16
    packed_latent_width = width // 16
    
    noise = torch.randn(1, packed_latent_height * packed_latent_width, 16 * 2 * 2, device=device, dtype=torch.float16)
    logger.debug("Noise prepared.")

    # Generate the image
    timesteps = flux_train_utils.get_schedule(20, noise.shape[1], shift=True)  # Sample steps = 20
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(device)

    t5_attn_mask = t5_attn_mask.to(device)
    ae_outputs = conditions[prompt]

    logger.debug("Image generation parameters set.")
    
    args = lambda: None
    args.frame_num = frame_num

    # clip_l.to("cpu")
    # t5xxl.to("cpu")

    model.to(device)

    print(f"Model device: {model.device}")
    print(f"Noise device: {noise.device}")
    print(f"Image IDs device: {img_ids.device}")
    print(f"T5 output device: {t5_out.device}")
    print(f"Text IDs device: {txt_ids.device}")
    print(f"L pooled device: {l_pooled.device}")

    # Run the denoising process
    with accelerator.autocast(), torch.no_grad():
        x = flux_train_utils.denoise(
            args, model, noise, img_ids, t5_out, txt_ids, l_pooled, timesteps=timesteps, guidance=1.0, t5_attn_mask=t5_attn_mask, ae_outputs=ae_outputs
        )
    logger.debug("Denoising process completed.")

    # Decode the final image
    x = x.float()
    x = flux_utils.unpack_latents(x, packed_latent_height, packed_latent_width)
    # model.to("cpu")
    ae.to(device)
    with accelerator.autocast(), torch.no_grad():
        x = ae.decode(x)
    logger.debug("Latents decoded into image.")
    # ae.to("cpu")

    # Convert the tensor to an image
    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    generated_image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    logger.info("Image generation completed.")
    return generated_image

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## FLUX Image Generation")

    with gr.Row():
        with gr.Column(scale=1):
            # Dropdown for selecting the recraft model
            recraft_model = gr.Dropdown(
                label="Select Recraft Model", 
                choices=["Wood Sculpture", "LEGO", "Sketch", "Portrait"], 
                value="Wood Sculpture"
            )
        
            # Load Model Button
            load_button = gr.Button("Load Model")

        with gr.Column(scale=1):
            # Status message box
            status_box = gr.Textbox(label="Status", placeholder="Model loading status", interactive=False, value="Model not loaded", lines=3)
    
    with gr.Row():
        with gr.Column(scale=0.5):
            # Input for the prompt
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=8)
            seed = gr.Slider(0, np.iinfo(np.int32).max, step=1, label="Seed", value=42)
        
        with gr.Column(scale=0.5):
            # File upload for image
            sample_image = gr.Image(label="Upload a Conditional Image", type="pil")
            run_button = gr.Button("Generate Image")
        
        with gr.Column(scale=1):
            # Output result
            result_image = gr.Image(label="Generated Image", interactive=False)

    # Load model button action
    load_button.click(fn=load_target_model, inputs=[recraft_model], outputs=[status_box])

    # Run Button 
    run_button.click(fn=infer, inputs=[prompt, sample_image, recraft_model, seed], outputs=[result_image])

    # Launch the Gradio app
    demo.launch()
