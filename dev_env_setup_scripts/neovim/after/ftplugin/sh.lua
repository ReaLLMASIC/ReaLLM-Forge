local opt = vim.opt_local

opt.expandtab = true
opt.shiftwidth = 2
opt.tabstop = 2
opt.softtabstop = 2
opt.autoindent = true
opt.smartindent = true

vim.bo.commentstring = "# %s"
vim.bo.makeprg = "bash %"

local bufnr = vim.api.nvim_get_current_buf()

vim.keymap.set("n", "<leader>rr", "<cmd>w | !bash %<CR>", { buffer = bufnr, desc = "Run current shell script" })

local has_toggleterm, toggleterm = pcall(require, "toggleterm.terminal")
if has_toggleterm then
  local Terminal = toggleterm.Terminal
  vim.keymap.set("n", "<leader>rs", function()
    local script = vim.api.nvim_buf_get_name(bufnr)
    if script == "" then
      vim.notify("Buffer must be saved before running in terminal", vim.log.levels.WARN)
      return
    end
    Terminal:new({
      cmd = string.format("bash %s", vim.fn.fnameescape(script)),
      close_on_exit = false,
      direction = "float",
    }):toggle()
  end, { buffer = bufnr, desc = "Run script in floating terminal" })
end
