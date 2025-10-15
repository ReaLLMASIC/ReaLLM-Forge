return {
  {
    "mfussenegger/nvim-dap",
    dependencies = {
      "rcarriga/nvim-dap-ui",
      "mfussenegger/nvim-dap-python",
      "theHamsta/nvim-dap-virtual-text",
      "jay-babu/mason-nvim-dap.nvim",
    },
    config = function()
      local dap = require("dap")
      local dapui = require("dapui")

      require("dapui").setup()
      require("nvim-dap-virtual-text").setup({
        virt_text_pos = "eol",
      })

      require("mason-nvim-dap").setup({
        ensure_installed = { "debugpy" },
        handlers = {},
      })

      dap.listeners.after.event_initialized["dapui_config"] = function()
        dapui.open()
      end
      dap.listeners.before.event_terminated["dapui_config"] = function()
        dapui.close()
      end
      dap.listeners.before.event_exited["dapui_config"] = function()
        dapui.close()
      end

      local python_host = vim.g.python3_host_prog and vim.fn.expand(vim.g.python3_host_prog) or vim.fn.exepath("python3")
      if python_host == nil or python_host == "" then
        python_host = "python3"
      end

      require("dap-python").setup(python_host, {
        include_configs = true,
      })
      require("dap-python").test_runner = "pytest"

      vim.keymap.set("n", "<F5>", function()
        dap.continue()
      end, { desc = "DAP Continue" })
      vim.keymap.set("n", "<F10>", function()
        dap.step_over()
      end, { desc = "DAP Step Over" })
      vim.keymap.set("n", "<F11>", function()
        dap.step_into()
      end, { desc = "DAP Step Into" })
      vim.keymap.set("n", "<F12>", function()
        dap.step_out()
      end, { desc = "DAP Step Out" })
      vim.keymap.set("n", "<leader>db", function()
        dap.toggle_breakpoint()
      end, { desc = "DAP Toggle Breakpoint" })
      vim.keymap.set("n", "<leader>dB", function()
        dap.set_breakpoint(vim.fn.input("Breakpoint condition: "))
      end, { desc = "DAP Conditional Breakpoint" })
      vim.keymap.set("n", "<leader>dr", function()
        dap.restart_frame()
      end, { desc = "DAP Restart Frame" })
      vim.keymap.set("n", "<leader>du", function()
        dapui.toggle({ reset = true })
      end, { desc = "Toggle DAP UI" })
    end,
  },
  {
    "akinsho/toggleterm.nvim",
    version = "*",
    config = function()
      require("toggleterm").setup({
        size = 20,
        open_mapping = [[<C-\>]],
        hide_numbers = true,
        shade_terminals = true,
        shading_factor = 2,
        direction = "float",
        float_opts = {
          border = "curved",
        },
      })
    end,
  },
  {
    "iamcco/markdown-preview.nvim",
    ft = { "markdown" },
    build = function()
      vim.fn["mkdp#util#install"]()
    end,
    config = function()
      vim.g.mkdp_auto_start = 0
      vim.g.mkdp_auto_close = 0
      vim.g.mkdp_browser = ""
      vim.g.mkdp_theme = "dark"
      vim.g.mkdp_filetypes = { "markdown" }
    end,
  },
  {
    "stevearc/aerial.nvim",
    dependencies = {
      "nvim-treesitter/nvim-treesitter",
      "nvim-tree/nvim-web-devicons",
    },
    config = function()
      require("aerial").setup({
        backends = { "lsp", "treesitter", "markdown" },
        layout = {
          default_direction = "prefer_right",
          placement = "edge",
        },
        filter_kind = false,
      })
      vim.keymap.set("n", "<leader>an", "<cmd>AerialNavOpen<CR>", { desc = "Open Aerial navigation" })
      vim.keymap.set("n", "<leader>aa", "<cmd>AerialToggle!<CR>", { desc = "Toggle Aerial outline" })
    end,
  },
  {
    "preservim/tagbar",
    cmd = { "TagbarToggle", "TagbarOpen" },
    init = function()
      vim.g.tagbar_autofocus = 1
      vim.g.tagbar_sort = 0
    end,
    config = function()
      vim.keymap.set("n", "<leader>tb", "<cmd>TagbarToggle<CR>", { desc = "Toggle Tagbar outline" })
    end,
  },
}
