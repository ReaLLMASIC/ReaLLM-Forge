local opt = vim.opt_local

opt.expandtab = true
opt.shiftwidth = 2
opt.tabstop = 2
opt.softtabstop = 2
opt.autoindent = true
opt.smartindent = true
opt.wrap = false
opt.colorcolumn = "80"

vim.bo.commentstring = "# %s"
vim.bo.makeprg = "yamllint %"

vim.keymap.set("n", "<leader>cf", function()
  vim.lsp.buf.format({ async = true })
end, { buffer = 0, desc = "Format YAML buffer" })
