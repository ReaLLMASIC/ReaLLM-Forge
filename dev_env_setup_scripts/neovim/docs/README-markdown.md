# Markdown Authoring

This Neovim profile includes tooling to preview, lint, and navigate Markdown documents efficiently.

## Previewing
- `iamcco/markdown-preview.nvim` provides a live browser preview.
- In a Markdown buffer press `<leader>mp` to toggle the preview window.
- The preview theme follows the dark UI; change it by setting `vim.g.mkdp_theme` in your `init.lua`.

## Navigation
- `<leader>mt` invokes `:Telescope treesitter`, showing a list of headings and symbols detected by Treesitter.
- `AerialToggle` (`<leader>aa`) opens an outline sidebar built from the same symbols.

## Editing Defaults
- Soft wrapping, line breaking, and spell checking are enabled automatically.
- Commenting uses the HTML style `<!-- comment -->` so that notes stay invisible in rendered output.
- Treesitter highlighting is installed for both `markdown` and `markdown_inline` for accurate fenced-code colors.

## Tips
- Use Gitsigns' hunk preview while reviewing docs to keep context and preview diffs inline.
- Combine UltiSnips with Markdown snippet collections (e.g. table skeletons) to accelerate formatting.
