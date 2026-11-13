# PR #926 · Char vs. CharBPE dashboard

Compare validation BPB and token-level diagnostics for Japanese, Korean, and Simplified Chinese in original text and IPA.

## Open the dashboard

Clone or pull the repository, then double-click `report/pr926-dashboard/index.html`
in your file manager. From the repository root on macOS:

```sh
open report/pr926-dashboard/index.html
```

Keep `index.html`, `assets/`, and `data/` together. The page runs offline without a server or build step. Each chart has a **Save PNG** button. Use **Paper evidence** in the page navigation for the comparison with Table 4 of the paper.

No Python, Node.js, package installation, CDN, dataset download, or running model
is needed to view it. Copying this entire directory to another location also
works. Use a modern browser with JavaScript and WebGL enabled for the interactive
token plots. Token datasets are loaded on demand through relative classic script
tags, so opening the page with `file://` does not require `fetch`, an HTTP server,
or cross-origin permissions. The two TSV files and the checkpoint CSV/PNG export
buttons are available locally.

## What these measurements represent

This is the supplied dashboard of the original TED-2020 training experiment,
including historical evaluations and the resumed ~3-epoch endpoints. Validation
used 50 sampled batches (819,200 target observations per checkpoint). It is not
the new full-corpus TED/FLORES evaluation, and it does not contain FLORES results.

The original layout, chart logic, Plotly bundle and numeric datasets are retained.
The page adds a visible snapshot label; this README and the repository entry
links explain how to open it. `provenance.json` identifies the supplied archive
and records SHA-256 hashes for the unchanged assets and datasets.

## Data and dependencies

- `data/` contains PR #926 evaluation summaries and per-checkpoint token labels and metrics for 12 runs. The paper comparison uses the summary coefficients from Table 4.
- `assets/plotly.min.js` is the Plotly.js 2.35.2 bundle. The MIT license shipped with that distribution is in `assets/plotly.LICENSE.txt`.
- Paper comparison coefficients are included as supplied in `data/token-evidence.js`.
- Checkpoint weights and the original tokenized training corpus are not needed
  to display any of these charts.

For new corpus measurements, see the
[full checkpoint evaluation guide](../../documentation/cjk_full_checkpoint_eval.md).
