return {
  {
    "tpope/vim-surround",
    keys = {
      { "cs", mode = "n" },
      { "ds", mode = "n" },
      { "ys", mode = { "n", "x" } },
      { "S", mode = "x" },
    },
    config = function()
      -- vim-surround works out of the box; key registrations above help lazy.nvim load on demand.
    end,
  },
}
