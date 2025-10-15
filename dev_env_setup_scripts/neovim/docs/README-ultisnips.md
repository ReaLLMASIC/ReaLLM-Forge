# UltiSnips and Completion Cheatsheet

The configuration keeps UltiSnips as the snippet engine behind `nvim-cmp`, while also enabling fast manual triggers.

## Triggers and Navigation
- `<Tab>` expands the snippet available at the cursor.
- `<C-j>` / `<C-k>` jump forward and backward between snippet placeholders.
- `<C-n>` in insert mode opens the completion menu. If a snippet is expandable in that position it takes precedence over other completion sources.
- `<leader>U` lists every UltiSnips snippet available for the current filetype so you can browse what's installed.

## Snippet Sources
- `honza/vim-snippets` ships with comprehensive Python, Markdown, YAML, and shell snippets.
- Place project-specific snippets in `~/.config/nvim/UltiSnips/<filetype>.snippets` and they will be loaded automatically.

## Mixing Snippets with Completion
- The completion menu shows snippet candidates with `[SNIP]` in the menu.
- Confirm a completion with `<CR>` or reuse `<Tab>` to expand the currently selected snippet entry.

## Recommended Workflow
1. Hit `<C-n>` to pull up completions.
2. Use `<Tab>` to expand the top UltiSnips suggestion.
3. Navigate placeholders with `<C-j>` / `<C-k>` and keep typing.

See `:help UltiSnips` for power features such as nested snippets and shell command interpolation.
