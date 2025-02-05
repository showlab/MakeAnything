#!/bin/bash

BASE_FLUX_CHECKPOINT="/path/to/output/merged_model.safetensors"
LORA_WEIGHTS_PATH="/path/to/recraft/recraft_model.safetensors"
OUTPUT_DIR="path/to/output_dir"

CLIP_L_PATH="path/to/clip_l.safetensors"
T5XXL_PATH="path/to/t5xxl.safetensors"
AE_PATH="path/to/vae.safetensors"

SAMPLE_IMAGES_FILE="path/to/sample_images.txt"
SAMPLE_PROMPTS_FILE="path/to/sample_prompts.txt"
frame_num=4     # 4 or 9



python flux_inference_recraft.py \
    --base_flux_checkpoint "$BASE_FLUX_CHECKPOINT" \
    --lora_weights_path "$LORA_WEIGHTS_PATH" \
    --clip_l_path "$CLIP_L_PATH" \
    --t5xxl_path "$T5XXL_PATH" \
    --ae_path "$AE_PATH" \
    --sample_images_file "$SAMPLE_IMAGES_FILE" \
    --sample_prompts_file "$SAMPLE_PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --frame_num $frame_num