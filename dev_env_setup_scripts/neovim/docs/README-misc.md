# Miscellaneous Enhancements

This document captures the remaining quality-of-life improvements bundled with the Neovim profile.

## Shell Scripting
- Shell buffers default to two-space indentation and expose `#` comments.
- `<leader>rr` saves and runs the current script with your system `bash`.
- `<leader>rs` executes the script inside a floating ToggleTerm window so you can observe long-running jobs.

## YAML Tooling
- Treesitter parses YAML by default and `yamlls` supplies completion plus schema-driven validation.
- `<leader>cf` formats the active buffer using the LSP server.
- `yamllint` is configured as the `makeprg`, making `:make` a quick validation shortcut.

## Markdown Extras
- Treesitter support for both block and inline Markdown ensures accurate highlighting in documentation-heavy files.
- Spell checking and word wrapping are enabled automatically for prose editing.

## Terminal Ergonomics
- `<leader>tt` / `<leader>th` / `<leader>tv` open floating, horizontal, and vertical ToggleTerm terminals.
- The default terminal mapping `<C-\>` is also available if you prefer to keep your hands on the home row.

## Surround Editing
- Powered by `tpope/vim-surround` for rapid quote/bracket manipulation.
- `ysiw"` wraps the inner word in double quotes, while `ds"` removes them.
- `cs"'` changes the surrounding double quotes into single quotes without retyping the text.
- Visual selections can be surrounded directly with `S"` / `S)` / `S]`, etc.

## Debugging Visuals
- `nvim-dap-ui` opens automatically when a debug session starts and closes once the session exits.
- `nvim-dap-virtual-text` annotates tensors and other variables inline while stepping through code.

## Where to Go Next
- `:help lazy.nvim.txt` explains how to manage plugins.
- `:Mason` and `:MasonInstall` let you manage additional LSP servers, formatters, and debuggers.
