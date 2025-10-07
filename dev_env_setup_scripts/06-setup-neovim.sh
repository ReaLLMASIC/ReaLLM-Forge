#!/bin/bash
#
# 06-setup-neovim.sh
#
# This script installs the latest Neovim and a modern, minimal Lua configuration
# focused on Python development with Catppuccin, Treesitter, Telescope, and UltiSnips.

set -e

# --- Helper Function for Logging ---
log() {
  GREEN='\033[0;32m'
  NC='\033[0m' # No Color
  echo -e "${GREEN}🚀 [$(date +'%T')] $1${NC}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVIM_TEMPLATE_DIR="${SCRIPT_DIR}/neovim"

# --- Installation ---
log "Installing Neovim and its configuration..."

# 1. Install Neovim (latest stable) via AppImage
log "Downloading latest Neovim AppImage..."
wget -O nvim.appimage https://github.com/neovim/neovim/releases/download/v0.11.4/nvim-linux-x86_64.appimage
chmod u+x nvim.appimage
mkdir -p ~/.local/bin
mv nvim.appimage ~/.local/bin/nvim
log "Neovim installed to ~/.local/bin/nvim"

# 2. Install dependencies
log "Installing npm and pip dependencies..."
python3 -m pip install pynvim

log "Installing dependencies for Telescope (ripgrep)..."
sudo apt-get update && sudo apt-get install -y ripgrep fd-find

# 3. Backup any existing Neovim configuration
NVIM_CONFIG_DIR="$HOME/.config/nvim"
if [ -d "$NVIM_CONFIG_DIR" ]; then
    log "Backing up existing Neovim config to ${NVIM_CONFIG_DIR}.bak..."
    mv "$NVIM_CONFIG_DIR" "${NVIM_CONFIG_DIR}.bak"
fi

# 4. Copy the repository-provided configuration
log "Creating new Lua-based Neovim configuration from templates..."
mkdir -p "$NVIM_CONFIG_DIR"
cp -r "${NVIM_TEMPLATE_DIR}/." "$NVIM_CONFIG_DIR"

# --- Final Message ---
echo ""
log "✅ Neovim setup is complete!"
log "Start Neovim by running: nvim"
log "Plugins will be installed automatically on the first run."
log "The Python Language Server (pyright) will also be installed automatically."
