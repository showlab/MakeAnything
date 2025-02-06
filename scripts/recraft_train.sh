#!/bin/bash

LORA_RANK=256
BATCH_SIZE=2
DATA_CONFIG="path/to/data_config.toml"
OUTPUT_DIR="path/to/output_dir"
OUTNAME="output_name"

LOG_DIR=$OUTPUT_DIR/logs
MAX_TRAIN_EPOCHS=100
MAX_TRAIN_STEPS=50000
SAVE_EVERY_N_EPOCHS=5
SAVE_EVERY_N_STEPS=5000

CKPT_PATH="path/to/unet_checkpoint.safetensors"
CLIP_L_PATH="path/to/clip_l.safetensors"
T5XXL_PATH="path/to/t5xxl.safetensors"
AE_PATH="path/to/vae.safetensors"

sample_images_path="path/to/sample_images.txt"
sample_prompts_path="path/to/sample_prompts.txt"
frame_num=9   # 4 for 1024 or 9 for 1056

accelerate launch --config_file="path/to/accelerate_config.yaml" \
  --main_process_port=23322 --mixed_precision=bf16 --num_cpu_threads_per_process=1 flux_train_recraft.py \
  --pretrained_model_name_or_path=$CKPT_PATH \
  --clip_l=$CLIP_L_PATH \
  --t5xxl=$T5XXL_PATH \
  --ae=$AE_PATH \
  --max_token_length=225 --apply_t5_attn_mask \
  --dataset_config=$DATA_CONFIG \
  --cache_latents --vae_batch_size=4 --cache_latents_to_disk --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --skip_cache_check \
  --enable_bucket --min_bucket_reso=256 --max_bucket_reso=2304 --bucket_reso_steps=64 --bucket_no_upscale \
  --output_dir=$OUTPUT_DIR --output_name=$OUTNAME --logging_dir=$LOG_DIR --log_with=tensorboard --log_prefix="" --log_tracker_name=tryon \
  --save_every_n_epochs=$SAVE_EVERY_N_EPOCHS --save_every_n_steps=$SAVE_EVERY_N_STEPS --save_precision=bf16 --save_model_as=safetensors \
  --max_train_epochs=$MAX_TRAIN_EPOCHS --max_train_steps=$MAX_TRAIN_STEPS \
  --initial_epoch=0 --initial_step=0 \
  --train_batch_size=$BATCH_SIZE --max_data_loader_n_workers=2 --persistent_data_loader_workers --mixed_precision=bf16 --fp8_base \
  --mem_eff_attn --sdpa --gradient_checkpointing --gradient_accumulation_steps=1 \
  --seed=42 --clip_skip=2 --noise_offset=0.0375 --loss_type=l2 --adaptive_noise_scale=0.0375 \
  --learning_rate=1 --unet_lr=1 --text_encoder_lr=1 \
  --optimizer_type=Prodigy --optimizer_args "weight_decay=0.01" "betas=.9,.99" "decouple=True" "use_bias_correction=True" "d_coef=0.5" "d0=1e-4" \
  --lr_scheduler=cosine_with_restarts --lr_scheduler_num_cycles=1 --lr_decay_steps=160 --lr_scheduler_min_lr_ratio=0.1 \
  --network_module=networks.lora_flux --network_dim=$LORA_RANK --network_alpha=$LORA_RANK --network_train_unet_only \
  --guidance_scale=1 --timestep_sampling=flux_shift --discrete_flow_shift 3.1582 \
  --model_prediction_type=raw \
  --sample_every_n_steps 300 \
  --sample_at_first \
  --sample_images $sample_images_path \
  --sample_prompts $sample_prompts_path \
  --frame_num $frame_num
