# Orin dev setup scripts

Scripts for turning a fresh Jetson Orin Nano into an LLM dev/inference
box — PyTorch, llama.cpp, Ollama, and the benchmarking harness.

## Setup, in order

| Script | Does |
|---|---|
| `00-setup-jetson-pytorch.sh` | Installs PyTorch/TorchVision/TorchAudio (Jetson AI Lab mirror, JetPack 6 / CUDA 12.6) + cuSPARSELt/cuDSS. Verifies CUDA at the end. |
| `01-setup-shakespeare-pip-dependencies.sh` | Python deps for the Shakespeare model experiments. Then: `bash data/shakespeare_char/get_dataset.sh` |
| `02-change-default-fan-curve.sh` | Installs `nvfancontrol.conf`, restarts the fan service, installs `jetson-stats` (`jtop`). |
| `03-setup-desktop-shortcuts.sh` | Adds **Run ReaLLM** and **Jetson Stats** desktop icons + background. |
| `04-swap-expand.sh` | Recreates 28GB swapfile, sets `vm.swappiness=100`. Safe to re-run. |
| `05-maxn-clocks.sh` | ⚠️ Locks power mode to MAXN + max clocks, **this session only**. |
| `06-jetson-clocks-service.sh` | ⚠️ Same as `05`, but as a **systemd service — persists across reboots**. |

> ⚠️ **`05`/`06` change the power profile.** If you're comparing performance
> or energy across two Orins, one having run these and the other not will
> fully explain a difference that looks like "hardware variance" but isn't.
> Check `nvpmodel -q` + `jtop`'s Jetson Clocks status on both before trusting
> a comparison.

## Other tools here

**`beta-jetson-config-v2.sh`** — `sudo`-run interactive TUI (whiptail) for
hostname, Wi-Fi/Bluetooth, power mode, fan profile, the clocks boot
service, 40-pin header config, and an AI Setup submenu. Auto-backs up any
file it touches to `<file>.bak`.

**`sweeper-bootstrap.sh`** — bootstrap for the separate LLM inference
*sweep harness* project (energy/latency/memory vs. prefill/decode/vocab
size). Installs everything that's missing — system deps, llama.cpp, the
model, the harness scripts. Sets up `~/Sweeper` as its project root.

**`llama.cpp/`**
- `07-Clone-Complie-Llama.cpp` — clone + CUDA build. *(Separate checkout
  from the one `sweeper-bootstrap.sh` manages — not required to match.)*
- `08-Llama.cpp-Puller` — menu picker for a few curated models, runs
  `llama-cli` directly against your choice.

**`ollama/`**
- `07-install-ollama` — official install script.
- `08-ollama-models` — scrapes + curates a model list, menu-driven `pull`.
- `09-install-openclaw` — Node 22 + launches `openclaw` via Ollama.
- `ollama-python/Working-python` — Python/Ollama usage scratch work.

**`camera-tests-python/`** — standalone OpenCV/PyTorch camera scripts
(ASCII render, grayscale, YOLOv8 pose, etc.) — run individually.

**`desktop/`, `pictures/`, `nvfancontrol.conf`** — assets for `02`/`03`
above. Not run directly.

## Resources
- [Setting up Pytorch on Jetson](https://medium.com/@surentharm/setting-up-pytorch-on-nvidia-jetson-nano-the-complete-2025-guide-294a7cf62766)
