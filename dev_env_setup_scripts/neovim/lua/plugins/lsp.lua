return {
  {
    "neovim/nvim-lspconfig",
    dependencies = {
      "williamboman/mason.nvim",
      "williamboman/mason-lspconfig.nvim",
      "b0o/schemastore.nvim",
    },
    config = function()
      require("mason").setup()

      local capabilities = require("cmp_nvim_lsp").default_capabilities()
      local lspconfig = require("lspconfig")

      local python_host = vim.g.python3_host_prog and vim.fn.expand(vim.g.python3_host_prog) or vim.fn.exepath("python3")
      if python_host == nil or python_host == "" then
        python_host = "python3"
      end
      local python_site_packages = vim.fn.expand("~/miniconda3/envs/reallmforge/lib/python3.10/site-packages")
      local python_extra_paths = {}
      if python_site_packages ~= "" and vim.loop.fs_stat(python_site_packages) then
        table.insert(python_extra_paths, python_site_packages)
      end

      local server_settings = {
        pyright = {
          settings = {
            python = {
              analysis = {
                autoImportCompletions = true,
                diagnosticMode = "workspace",
                typeCheckingMode = "basic",
                useLibraryCodeForTypes = true,
                extraPaths = python_extra_paths,
              },
              pythonPath = python_host,
            },
          },
        },
        ruff_lsp = {
          init_options = {
            settings = {
              args = {},
            },
          },
        },
        yamlls = {
          settings = {
            yaml = {
              format = { enable = true },
              validate = true,
              schemaStore = {
                enable = false,
                url = "",
              },
              schemas = require("schemastore").yaml.schemas(),
            },
          },
        },
        marksman = {},
        bashls = {},
      }

      require("mason-lspconfig").setup({
        ensure_installed = {
          "pyright",
          "ruff_lsp",
          "yamlls",
          "marksman",
          "bashls",
        },
        handlers = {
          function(server_name)
            local opts = { capabilities = capabilities }
            if server_settings[server_name] then
              opts = vim.tbl_deep_extend("force", opts, server_settings[server_name])
            end
            lspconfig[server_name].setup(opts)
          end,
        },
      })

      vim.keymap.set("n", "K", vim.lsp.buf.hover, { desc = "LSP Hover" })
      vim.keymap.set("n", "gd", vim.lsp.buf.definition, { desc = "LSP Definition" })
      vim.keymap.set("n", "<leader>ca", vim.lsp.buf.code_action, { desc = "LSP Code Action" })
    end,
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

      local function feedkeys(key)
        vim.api.nvim_feedkeys(vim.api.nvim_replace_termcodes(key, true, true, true), "", true)
      end

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
          ['<C-n>'] = cmp.mapping(function(fallback)
            if cmp.visible() then
              cmp.select_next_item()
            else
              cmp.complete()
            end
          end, { 'i', 's' }),
          ['<Tab>'] = cmp.mapping(function(fallback)
            if vim.fn["UltiSnips#CanExpandSnippet"]() == 1 then
              feedkeys("<Plug>(ultisnips-expand)")
            elseif vim.fn["UltiSnips#CanJumpForwards"]() == 1 then
              feedkeys("<Plug>(ultisnips-jump-forwards)")
            elseif cmp.visible() then
              cmp.confirm({ select = true })
            else
              fallback()
            end
          end, { 'i', 's' }),
          ['<S-Tab>'] = cmp.mapping(function(fallback)
            if vim.fn["UltiSnips#CanJumpBackwards"]() == 1 then
              feedkeys("<Plug>(ultisnips-jump-backwards)")
            elseif cmp.visible() then
              cmp.select_prev_item()
            else
              fallback()
            end
          end, { 'i', 's' }),
        }),
      })
    end
  }
}
