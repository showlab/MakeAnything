#!/bin/bash

CKPT_PATH="/path/to/unet/flux1-dev.safetensors"
CLIP_L_PATH="/path/to/clip/clip_l.safetensors"
T5XXL_PATH="/path/to/clip/t5xxl_fp16.safetensors"
AE_PATH="/path/to/vae/ae.safetensors"

dataset_config="/path/to/your/dataset/config.toml" 
output_dir="/path/to/output/directory"
output_name="Model_Name"

lora_ups_num=10
network_dim=64
max_train_steps=50000

accelerate launch \
  --config_file "/path/to/accelerate/config.yaml" \
  --num_cpu_threads_per_process 1 \
  --gpu_ids 1 \
  flux_train_network_asylora.py \
  --dataset_config $dataset_config \
  --pretrained_model_name_or_path $CKPT_PATH \
  --ae $AE_PATH \
  --clip_l $CLIP_L_PATH \
  --t5xxl $T5XXL_PATH \
  --optimizer_type came \
  --max_grad_norm 1.0 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --lr_scheduler_num_cycles 1 \
  --lr_scheduler_power 1.0 \
  --min_snr_gamma 5 \
  --output_name $output_name \
  --output_dir $output_dir \
  --network_dim $network_dim \
  --network_alpha 1.0 \
  --learning_rate 1e-4 \
  --max_train_steps $max_train_steps \
  --apply_t5_attn_mask \
  --cache_latents_to_disk \
  --cache_text_encoder_outputs \
  --cache_text_encoder_outputs_to_disk \
  --weighting_scheme logit_normal \
  --logit_mean 0 \
  --logit_std 1.0 \
  --mode_scale 1.29 \
  --timestep_sampling shift \
  --sigmoid_scale 1.0 \
  --model_prediction_type raw \
  --guidance_scale 1.0 \
  --discrete_flow_shift 3.1582 \
  --fp8_base \
  --highvram \
  --gradient_checkpointing \
  --seed 42 \
  --save_precision bf16 \
  --save_every_n_epochs 5 \
  --network_module networks.asylora_flux \
  --network_train_unet_only \
  --vae_batch_size 1 \
  --save_model_as safetensors \
  --max_data_loader_n_workers 0 \
  --mixed_precision bf16 \
  --skip_cache_check \
  --gradient_accumulation_steps 1 \
  --lora_ups_num $lora_ups_num \
  --log_config