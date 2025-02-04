#!/bin/bash

CKPT_PATH="/path/to/unet/flux1-dev.safetensors"
CLIP_L_PATH="/path/to/clip/clip_l.safetensors"
T5XXL_PATH="/path/to/clip/t5xxl_fp16.safetensors"
AE_PATH="/path/to/vae/ae.safetensors"
LORA_PATH="/path/to/asylora/asylora.safetensors"
OUTPUT_DIR="/path/to/output/directory"


# Number of B matrices used in asymmetric LoRA.
lora_ups_num=10
# The current B matrix index to be used. Specify the matrix index you want to apply.
lora_up_cur=8

PROMPT="YOUR_PROMPT"

python flux_minimal_inference_asylora.py \
  --ckpt_path $CKPT_PATH \
  --clip_l $CLIP_L_PATH \
  --t5xxl $T5XXL_PATH \
  --ae $AE_PATH \
  --prompt "$PROMPT" \
  --width 1056 \
  --height 1056 \
  --steps 25 \
  --dtype bf16 \
  --output_dir $OUTPUT_DIR \
  --flux_dtype fp8 \
  --offload \
  --lora_ups_num $lora_ups_num \
  --lora_up_cur $lora_up_cur \
  --lora_weights $LORA_PATH
