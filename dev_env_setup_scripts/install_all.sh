#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

echo "Note, these scripts can override local dotfiles, and are intended for use
with newly instantiated VMs, and not tested for existing setups."
echo "Press ctrl-c or equivalent to quit now, or press enter to begin setup."
read okay
#
# --- Helper Function for Logging ---
log() {
  # ANSI color codes
  GREEN='\033[0;32m'
  NC='\033[0m' # No Color
  echo -e "${GREEN}🚀 [$(date +'%T')] $1${NC}"
}

# --- Main Setup ---
log "Starting the full machine setup..."

log "Step 0: Setting up system packages..."
bash ./00-setup-conda.sh

log "Activating conda environment..."
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate reallmforge
else
  echo "Unable to locate conda initialization script at $CONDA_SH"
  exit 1
fi

log "Step 1: Setting up Zsh..."
bash ./01-setup-zsh.sh

log "Step 2: Setting up Tmux..."
bash ./02-setup-tmux.sh

log "Step 3: Setting up node..."
bash ./03-setup-node.sh

log "Step 4: Setting up Rust..."
bash ./04-setup-rust.sh

log "Step 5: Setting up Tectonic..."
bash ./05-setup-tectonic.sh

log "Step 6: Setting up Neovim..."
bash ./06-setup-neovim.sh

log "Step 7: Setting up bash_aliases..."
bash ./07-setup-bash-aliases.sh

log "Step 8: Setting up gh cli util..."
bash ./08-setup-gh-cli-util.sh

log "Step 9: Setting up global github email and default name..."
bash ./09-setup-git-user.sh

echo ""
log "✅ All setup scripts executed successfully!"
log "Please log out and log back in for all changes to take effect."
