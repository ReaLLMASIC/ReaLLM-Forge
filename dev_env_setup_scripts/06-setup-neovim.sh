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

# 4. Create the new configuration directories and files
log "Creating new Lua-based Neovim configuration..."
mkdir -p ~/.config/nvim/lua/core
mkdir -p ~/.config/nvim/lua/plugins
mkdir -p ~/.config/nvim/after/plugin

# --- Write the config files ---

# Write init.lua (Main Entry Point)
cat > ~/.config/nvim/init.lua << 'EOF'
-- Set the leader key BEFORE loading any plugins
vim.g.mapleader = ","
vim.g.maplocalleader = ","

-- Bootstrap lazy.nvim plugin manager
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
  vim.fn.system({
    "git",
    "clone",
    "--filter=blob:none",
    "https://github.com/folke/lazy.nvim.git",
    "--branch=stable",
    lazypath,
  })
end
vim.opt.rtp:prepend(lazypath)

-- Load core settings
require("core.options")
require("core.keymaps")

-- Load plugins
require("lazy").setup("plugins")
EOF

# Write lua/core/options.lua (Basic Settings)
cat > ~/.config/nvim/lua/core/options.lua << 'EOF'
local opt = vim.opt -- for conciseness

-- Line Numbers
opt.relativenumber = true
opt.number = true

-- Tabs and Indentation
opt.tabstop = 4
opt.shiftwidth = 4
opt.expandtab = true
opt.autoindent = true

-- Search
opt.ignorecase = true
opt.smartcase = true

-- Appearance
opt.termguicolors = true
opt.signcolumn = "yes"
opt.colorcolumn = "80"

-- Behavior
opt.clipboard = "unnamedplus" -- Sync with system clipboard
opt.splitright = true
opt.splitbelow = true
opt.mouse = "a"

-- Python host program
vim.g.python3_host_prog = '~/miniconda3/envs/reallmforge/bin/python3'
EOF

# Write lua/core/keymaps.lua (Essential Keymaps)
cat > ~/.config/nvim/lua/core/keymaps.lua << 'EOF'
local keymap = vim.keymap

-- General Keymaps
keymap.set("n", "<leader>sv", "<C-w>v", { desc = "Split window vertically" })
keymap.set("n", "<leader>sh", "<C-w>s", { desc = "Split window horizontally" })
keymap.set("n", "<leader>se", "<C-w>=", { desc = "Make splits equal size" })
keymap.set("n", "<leader>sx", "<cmd>close<CR>", { desc = "Close current split" })

-- Telescope Keymaps
keymap.set('n', '<leader>ff', '<cmd>Telescope find_files<cr>', { desc = 'Fuzzy find files' })
keymap.set('n', '<leader>fg', '<cmd>Telescope live_grep<cr>', { desc = 'Live grep' })
keymap.set('n', '<leader>fb', '<cmd>Telescope buffers<cr>', { desc = 'Find buffers' })
keymap.set('n', '<leader>fh', '<cmd>Telescope help_tags<cr>', { desc = 'Find help tags' })

-- Ultisnips
vim.g.UltiSnipsExpandTrigger = "<tab>"
vim.g.UltiSnipsJumpForwardTrigger = "<c-j>"
vim.g.UltiSnipsJumpBackwardTrigger = "<c-k>"
EOF

# --- Plugin Configurations ---

# Write lua/plugins/theme.lua (Catppuccin)
cat > ~/.config/nvim/lua/plugins/theme.lua << 'EOF'
return {
  "catppuccin/nvim",
  name = "catppuccin",
  priority = 1000,
  config = function()
    require("catppuccin").setup({
      flavour = "mocha",
    })
    vim.cmd.colorscheme "catppuccin"
  end,
}
EOF

# Write lua/plugins/telescope.lua
cat > ~/.config/nvim/lua/plugins/telescope.lua << 'EOF'
return {
  'nvim-telescope/telescope.nvim',
  tag = '0.1.5',
  dependencies = { 'nvim-lua/plenary.nvim' }
}
EOF

# Write lua/plugins/treesitter.lua
cat > ~/.config/nvim/lua/plugins/treesitter.lua << 'EOF'
return {
  "nvim-treesitter/nvim-treesitter",
  build = ":TSUpdate",
  config = function()
    require("nvim-treesitter.configs").setup({
      ensure_installed = { "c", "lua", "vim", "vimdoc", "python", "bash" },
      sync_install = false,
      auto_install = true,
      highlight = {
        enable = true,
      },
    })
  end,
}
EOF

# Write lua/plugins/lsp.lua (LSP, Completion, and Snippets)
cat > ~/.config/nvim/lua/plugins/lsp.lua << 'EOF'
return {
  {
    "neovim/nvim-lspconfig",
    dependencies = {
      "williamboman/mason.nvim",
      "williamboman/mason-lspconfig.nvim",
    },
    config = function()
      -- Set up mason and mason-lspconfig
      require("mason").setup()

      local capabilities = require('cmp_nvim_lsp').default_capabilities()

      require("mason-lspconfig").setup({
        ensure_installed = { "pyright" },
        handlers = {
          -- The first handler setup is for the servers that mason-lspconfig manages
          function(server_name)
            require("lspconfig")[server_name].setup({
              capabilities = capabilities,
            })
          end,
        },
      })

      -- Keymaps for LSP (these are fine)
      vim.keymap.set('n', 'K', vim.lsp.buf.hover, { desc = 'LSP Hover' })
      vim.keymap.set('n', 'gd', vim.lsp.buf.definition, { desc = 'LSP Definition' })
      vim.keymap.set('n', '<leader>ca', vim.lsp.buf.code_action, { desc = 'LSP Code Action' })
    end
  },
  {
    "hrsh7th/nvim-cmp",
    dependencies = {
      "hrsh7th/cmp-nvim-lsp",
      "quangnguyen30192/cmp-nvim-ultisnips",
      "SirVer/ultisnips",
      "honza/vim-snippets"
    },
    config = function()
      local cmp = require('cmp')
      cmp.setup({
        snippet = {
          expand = function(args)
            vim.fn["UltiSnips#Anon"](args.body)
          end,
        },
        sources = {
          { name = 'nvim_lsp' },
          { name = 'ultisnips' },
        },
        mapping = cmp.mapping.preset.insert({
          ['<C-Space>'] = cmp.mapping.complete(),
          ['<C-e>'] = cmp.mapping.abort(),
          ['<CR>'] = cmp.mapping.confirm({ select = true }),
        }),
      })
    end
  }
}
EOF

# --- Final Message ---
echo ""
log "✅ Neovim setup is complete!"
log "Start Neovim by running: nvim"
log "Plugins will be installed automatically on the first run."
log "The Python Language Server (pyright) will also be installed automatically."
