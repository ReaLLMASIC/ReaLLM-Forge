# export_quantization_data_to_pkl.py

from export_weights.loader import load_checkpoint
from export_weights.exporters import export_to_pkl
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export quantized weights/activations to PKL')
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1')")
    parser.add_argument('--out_dir', type=str, default='out', help='Directory to load checkpoint from')
    parser.add_argument('--file_name', type=str, default='quantized_weights', help='File name to export the quantized weights/activations')
    args = parser.parse_args()

    state_dict = load_checkpoint(args.out_dir, args.device)
    export_to_pkl(state_dict, args.file_name)
