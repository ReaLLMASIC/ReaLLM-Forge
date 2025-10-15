local opt = vim.opt_local

opt.expandtab = true
opt.shiftwidth = 4
opt.tabstop = 4
opt.softtabstop = 4
opt.textwidth = 88
opt.colorcolumn = "88"
opt.foldmethod = "indent"
opt.foldlevel = 99

vim.bo.commentstring = "# %s"
vim.bo.makeprg = "pytest %"

local has_dap_python, dap_python = pcall(require, "dap-python")
if has_dap_python then
  local bufnr = vim.api.nvim_get_current_buf()
  vim.keymap.set("n", "<leader>tm", function()
    dap_python.test_method()
  end, { buffer = bufnr, desc = "Run Pytest on current method" })
  vim.keymap.set("n", "<leader>tc", function()
    dap_python.test_class()
  end, { buffer = bufnr, desc = "Run Pytest on current class" })
  vim.keymap.set("n", "<leader>tf", function()
    dap_python.debug_selection()
  end, { buffer = bufnr, desc = "Debug selection with debugpy" })
end

local has_aerial, aerial = pcall(require, "aerial")
if has_aerial then
  aerial.open({ focus = false })
end
