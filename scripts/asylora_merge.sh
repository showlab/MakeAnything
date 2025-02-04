#!/bin/bash

python networks/flux_merge_lora.py \
  --flux_model "/path/to/flux_model/flux1-dev.safetensors" \
  --save_to "/path/to/output/merged_model.safetensors" \
  --models "/path/to/lora/lora.safetensors" \
  --ratios 1
