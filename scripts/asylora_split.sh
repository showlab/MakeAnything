#!/bin/bash

ASYLORA_PATH="/path/to/your/input_file.safetensors"
OUTPUT_PATH="/path/to/your/output_file.safetensors"
LORA_UP=5   # specified lora_up num

python split_asylora.py "$ASYLORA_PATH" "$OUTPUT_PATH" "$LORA_UP"
