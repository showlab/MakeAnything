import argparse
import os
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asylora_path', type=str, required=True, help="Path to the input asylora file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the modified safetensors file.")
    parser.add_argument('--lora_up', type=int, required=True, help="The target lora_up value.")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with safe_open(args.asylora_path, framework="pt") as f:
        tensor_dict = {key: f.get_tensor(key) for key in f.keys()}

        modified_dict = {}

        for key, tensor in tensor_dict.items():
            if 'lora_ups' in key:
                lora_up_index = int(key.split('.')[2])
                if lora_up_index != args.lora_up - 1:
                    continue
                else:
                    new_key = key.replace(f'lora_ups.{lora_up_index}.', 'lora_up.')
                    modified_dict[new_key] = tensor
            else:
                modified_dict[key] = tensor

        save_file(modified_dict, args.output_path)

if __name__ == "__main__":
    main()
