local opt = vim.opt_local

opt.wrap = true
opt.linebreak = true
opt.spell = true
opt.spelllang = { "en_us" }
opt.conceallevel = 2
opt.textwidth = 0
opt.colorcolumn = ""

vim.bo.commentstring = "<!-- %s -->"

vim.keymap.set("n", "<leader>mp", "<cmd>MarkdownPreviewToggle<CR>", { buffer = 0, desc = "Toggle Markdown preview" })
vim.keymap.set("n", "<leader>mt", "<cmd>Telescope treesitter<CR>", { buffer = 0, desc = "List document symbols" })

local has_aerial, aerial = pcall(require, "aerial")
if has_aerial then
  aerial.open({ focus = false })
end
