# Python & PyTorch Workflow

The setup is tuned for day-to-day Python and PyTorch development with strong navigation, completion, and debugging defaults.

## Language Servers & Formatting
- `pyright` powers completions and diagnostics with workspace-wide analysis and PyTorch-friendly import discovery.
- `ruff_lsp` provides fast linting feedback; format with `:lua vim.lsp.buf.format()` to run your configured server-side formatter.
- Extra paths from the Conda `reallmforge` environment are injected automatically when present so PyTorch packages are resolvable.

## Navigation
- `<leader>ff` / `<leader>fg` search files or ripgrep the project with Telescope.
- `<leader>aa` toggles Aerial's outline for jump-to-symbol navigation.
- `<leader>tb` opens Tagbar's class/method tree; press `<CR>` on any symbol to jump there or `p` to preview.
- Motion cheatsheet for Python buffers:

  | Mapping        | Purpose                                   |
  | -------------- | ------------------------------------------ |
  | `]]` / `[[`    | Jump to next / previous top-level class    |
  | `]m` / `[m`    | Jump to next / previous function or method |
  | `]M` / `[M`    | Jump to next / previous class definition   |
  | `gd`           | Go to definition via LSP                   |
  | `gD`           | Go to declaration                          |
  | `<leader>ca`   | Trigger LSP code actions                   |

- Use `gr` to list references in Telescope and `<leader>rn` to rename the symbol under the cursor.

## Debugging & Testing
- `<F5>`, `<F10>`, `<F11>`, and `<F12>` control debugpy sessions through `nvim-dap`.
- `<leader>db` toggles breakpoints, while `<leader>dB` adds a conditional breakpoint.
- Inside Python buffers, UltiSnips exposes PyTorch-friendly snippets; run targeted tests with `<leader>tm` (method) or `<leader>tc` (class).
- `<leader>tf` starts an interactive debug session on the visually selected block—perfect for stepping through tensor code.

## Terminals & REPLs
- `<leader>tt` spawns a floating terminal for quick experiments (ideal for `ipython`).
- `<leader>th` / `<leader>tv` open split terminals; use them with `python -m torch.utils.bottleneck` or other CLI tooling.

## Recommended Extensions
- Add project-specific snippets under `UltiSnips/python.snippets` for recurring PyTorch module templates.
- Use `:Mason` to install optional tools such as `black`, `isort`, or `debugpy` upgrades.
