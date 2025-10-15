return {
  {
    "tpope/vim-fugitive",
    cmd = { "Git", "G", "Gdiffsplit", "Gread", "Gwrite", "Ggrep", "GBrowse" },
    config = function()
      vim.keymap.set("n", "<leader>gs", "<cmd>Git<CR>", { desc = "Open Fugitive status" })
    end,
  },
  {
    "lewis6991/gitsigns.nvim",
    config = function()
      require("gitsigns").setup({
        on_attach = function(bufnr)
          local map = function(mode, lhs, rhs, desc)
            vim.keymap.set(mode, lhs, rhs, { buffer = bufnr, desc = desc })
          end

          map("n", "<leader>gh", require("gitsigns").stage_hunk, "Stage hunk")
          map("n", "<leader>gH", require("gitsigns").stage_buffer, "Stage buffer")
          map("n", "<leader>gu", require("gitsigns").undo_stage_hunk, "Undo stage hunk")
          map("n", "<leader>gr", require("gitsigns").reset_hunk, "Reset hunk")
          map("n", "<leader>gR", require("gitsigns").reset_buffer, "Reset buffer")
          map("n", "<leader>gd", require("gitsigns").diffthis, "Diff current file")
          map("n", "<leader>gp", require("gitsigns").preview_hunk, "Preview hunk")
          map("n", "<leader>gt", require("gitsigns").toggle_current_line_blame, "Toggle current line blame")
          map("n", "]c", require("gitsigns").next_hunk, "Next hunk")
          map("n", "[c", require("gitsigns").prev_hunk, "Previous hunk")
        end,
      })
    end,
  },
  {
    "TaDaa/vimade",
    config = function()
      vim.g.vimade = vim.g.vimade or {}
    end,
  },
}
