import argparse
import os
import torch
import torch.nn as nn
import numpy as np

from model import GPT
from gpt_conf import GPTConfig


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu')
    config = GPTConfig(**ckpt['model_args'])
    model = GPT(config)
    state_dict = ckpt['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model, ckpt


def get_calib_data(data_path, block_size, batch_size, num_batches):
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    # trim to multiples of batch_size*block_size
    length = batch_size * block_size * num_batches
    data = torch.from_numpy(data[:length].astype(np.int64))
    data = data.view(num_batches, batch_size, block_size)
    return data


def collect_activation_stats(model, calib_data, device='cpu'):
    stats = {}

    def get_hook(name):
        def hook(module, inp, output):
            x = inp[0].to(device)
            # B,T,C -> C
            cur_max = x.abs().amax(dim=(0, 1))
            if name in stats:
                stats[name] = torch.maximum(stats[name], cur_max)
            else:
                stats[name] = cur_max
        return hook

    handles = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(get_hook(n)))

    model.eval()
    with torch.no_grad():
        for batch in calib_data:
            batch = batch.to(device)
            model(batch)

    for h in handles:
        h.remove()
    return stats


def apply_smoothquant(model, act_stats, alpha=0.5):
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in act_stats:
            continue
        x_max = act_stats[name]
        w = module.weight.data
        w_max = w.abs().amax(dim=1)
        s = (x_max.pow(alpha) / (w_max + 1e-5).pow(1 - alpha))
        module.weight.data = w * s.unsqueeze(1)
        module.register_buffer('smoothquant_scale', s)
    return model


def main(args):
    model, ckpt = load_checkpoint(args.ckpt)
    calib = get_calib_data(args.data, ckpt['model_args']['block_size'], args.batch_size, args.calib_batches)
    act_stats = collect_activation_stats(model, calib, device=args.device)
    apply_smoothquant(model, act_stats, alpha=args.alpha)
    ckpt['model'] = model.state_dict()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'ckpt_smoothquant.pt')
    torch.save(ckpt, out_path)
    print(f"SmoothQuant checkpoint saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply SmoothQuant to a checkpoint')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint to load')
    parser.add_argument('--data', type=str, required=True, help='path to train.bin for calibration')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--calib_batches', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out_dir', type=str, default='smoothquant_ckpt')
    args = parser.parse_args()
    main(args)
