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
