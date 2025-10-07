local keymap = vim.keymap

-- General Keymaps
keymap.set("n", "<leader>sv", "<C-w>v", { desc = "Split window vertically" })
keymap.set("n", "<leader>sh", "<C-w>s", { desc = "Split window horizontally" })
keymap.set("n", "<leader>se", "<C-w>=", { desc = "Make splits equal size" })
keymap.set("n", "<leader>sx", "<cmd>close<CR>", { desc = "Close current split" })

-- Terminals
keymap.set("n", "<leader>tt", "<cmd>ToggleTerm<CR>", { desc = "Toggle floating terminal" })
keymap.set("n", "<leader>th", "<cmd>ToggleTerm direction=horizontal<CR>", { desc = "Toggle horizontal terminal" })
keymap.set("n", "<leader>tv", "<cmd>ToggleTerm direction=vertical<CR>", { desc = "Toggle vertical terminal" })

-- Telescope Keymaps
keymap.set('n', '<leader>ff', '<cmd>Telescope find_files<cr>', { desc = 'Fuzzy find files' })
keymap.set('n', '<leader>fg', '<cmd>Telescope live_grep<cr>', { desc = 'Live grep' })
keymap.set('n', '<leader>fb', '<cmd>Telescope buffers<cr>', { desc = 'Find buffers' })
keymap.set('n', '<leader>fh', '<cmd>Telescope help_tags<cr>', { desc = 'Find help tags' })

-- Ultisnips
vim.g.UltiSnipsExpandTrigger = "<tab>"
vim.g.UltiSnipsJumpForwardTrigger = "<c-j>"
vim.g.UltiSnipsJumpBackwardTrigger = "<c-k>"
keymap.set(
  "n",
  "<leader>U",
  "<cmd>UltiSnipsListSnippets<CR>",
  { desc = "List UltiSnips snippets for current filetype" }
)
