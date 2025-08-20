# Sliding Window with Flex Attention

Flex attention helps save manual implementations of sliding windows, providing memory efficiency and faster runtime performance.

Note: at time of writing "compile" is required for obtaining the performance improvements with flex_attention.

---

## Requirements

- **PyTorch** >= 2.5.0
- **pip**: Latest version recommended

---

## Upgrading PyTorch on Existing Setups

If your PyTorch is less than 2.5.0, follow the following steps to upgrade necessary dependencies:

```bash
pip3 install --upgrade pip
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Example Invocation 

At time of writing, in order to obtain the efficiency gains from flex-attention you must include the `--compile` flag in train.py invocations.

```bash
python3 train.py --compile --use_flex_attn --window_size 128 --block_size 1024 --disable_flash_attention
```
