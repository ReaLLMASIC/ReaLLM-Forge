#!/bin/bash
#
# This script installs tmux and its dependencies, installs the TPM plugin manager,
# and copies a local tmux.conf file into the home directory.

echo "🚀 Starting Tmux setup..."

# ------------------------------------------------------------------------------
# STEP 1: Install System Dependencies (tmux, git, xclip)
# ------------------------------------------------------------------------------
echo "📦 Installing required packages: tmux, git, and xclip..."
sudo apt-get update
sudo apt-get install -y tmux git xclip
echo "✅ Packages installed."

# ------------------------------------------------------------------------------
# STEP 2: Install Tmux Plugin Manager (TPM)
# ------------------------------------------------------------------------------
TPM_DIR="$HOME/.tmux/plugins/tpm"
if [ ! -d "$TPM_DIR" ]; then
    echo " installing Tmux Plugin Manager (TPM)..."
    git clone https://github.com/tmux-plugins/tpm "$TPM_DIR"
    echo "✅ TPM installed."
else
    echo "ℹ️ TPM is already installed."
fi

# ------------------------------------------------------------------------------
# STEP 3: Copy the local tmux.conf file
# ------------------------------------------------------------------------------
CONFIG_SOURCE="./tmux.conf"

if [ -f "$CONFIG_SOURCE" ]; then
    echo "⚙️  Copying local ./tmux.conf to ~/.tmux.conf..."
    cp "$CONFIG_SOURCE" ~/.tmux.conf
    echo "✅ Configuration file copied."
else
    echo "❌ Error: 'tmux.conf' not found in the current directory."
    echo "   Please create the file and place it alongside this script."
    exit 1
fi

echo ""
echo "✨ All done! Please follow these final steps:"
echo ""
echo "  1. Start tmux by running: tmux"
echo ""
echo "  2. Once inside tmux, press 'Ctrl + a' then 'I' (capital i) to fetch the plugins."
echo ""
echo "  3. The status bar at the bottom should change and show the powerline theme."
echo "     (NOTE: For powerline icons to work, your terminal must be using a Nerd Font)."
