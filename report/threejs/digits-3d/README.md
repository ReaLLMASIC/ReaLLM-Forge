# 3D digit-token trajectory viewer

This viewer follows the single-file Three.js report style in this directory,
including OrbitControls, canvas labels, a dark explanatory panel, and keyboard
controls. Width-2 runs are displayed natively in the XY plane, and width-3 runs
show native coordinates. Higher-dimensional runs use a
single PCA basis fitted across every token and checkpoint, rather than fitting
each frame independently, so motion remains comparable over time.

Run `demos/digits_3d_trajectory_demo.sh` from the repository root to generate
`token_trajectories.json`. Then serve the repository with
`python3 -m http.server 8000` and visit
`http://localhost:8000/report/threejs/digits-3d/index.html`.

Orange points are trained digit-like tokens. Blue points are configurable
letters that exist in the vocabulary and model parameters but never occur in
either dataset split. Use the slider, arrow keys, or Space playback to compare
their trajectories over checkpoint time.

By default embeddings remain on the radius-`sqrt(EMBEDDING_DIM)` sphere, so
motion in model space is directional. A radius guide is shown as a circle for
native 2D runs and a wireframe sphere for native 3D runs because PCA does not
preserve individual vector norms. Run the
demo with `WTE_FIXED_NORM=false` to produce the unconstrained view.

The underlying `viewer.html` also accepts `?data=<relative-json-path>`. The sweep demo uses this
to keep every vocabulary-size and fixed-norm variation under `runs/` rather
than overwriting the default trajectory.

The default sweep uses both 2 and 3 dimensions, 10 trained symbols, 10 held-out letters,
and both tied and untied WTE/LM-head runs, with 10,000 iterations per
configuration. Set `EMBEDDING_DIMS` (for example, `2 3 8 16 64`),
`DIGIT_COUNTS`, `LETTER_COUNTS`, `WTE_TYING_MODES`, or `SWEEP_MAX_ITERS` to
customize it.

After the first sweep run completes, open `index.html` to filter and select the
generated JSON files without editing URLs. Filters and the selected run are
stored in the URL query string, so a configured view can be bookmarked or shared.
Each trajectory includes checkpoint
train and validation loss; the viewer plots both above the time slider and
marks the currently selected checkpoint.

The sweep rewrites `runs/manifest.json` atomically after every completed run.
You can therefore keep the HTTP server open and refresh `index.html` while the
remaining configurations continue training.

The base sweep runs 30,000 iterations. It covers sudden included-to-excluded
and excluded-to-included transitions at 20%, 40%, 60%, and 80%, plus PWM-style
duty cycles whose included fraction is 20%, 40%, 60%, or 80% of each period.
`DROPOUT_COUNTS` accepts multiple counts to affect several trailing symbols at
once; `TRANSITION_PERCENTAGES`, `DUTY_CYCLES`, and `DUTY_PERIOD_PERCENT`
customize the schedules. Set `DUTY_PERIOD_PERCENT=0` to omit duty-cycle runs
while retaining the one-time drop/add schedules. Excluded points and trajectory segments are purple;
included segments retain the trained-token orange, including every transition
back to the dataset. Every schedule phase saves an end-of-phase checkpoint, so
transitions shorter than `SWEEP_SAVE_INTERVAL` can still resume correctly.

Every base-sweep schedule also compares full Muon with zero weight decay, Adam
with weight decays 0.0, 0.01, 0.05, 0.1, and 0.5, plus Adagrad, SGD, and RMSprop.
The latter three use zero weight decay. Set `OPTIMIZER_MODES` to choose a subset
or `ADAM_WEIGHT_DECAYS` to change the Adam sweep. Optimizer and effective weight decay are stored
in each trajectory, displayed on the run card, and available as a selector
filter.

The sweep includes unconstrained (`free`), `sqrt_dim`, and radius `1` runs by
default. Set `RADIUS_MODES`, for example `RADIUS_MODES="free sqrt_dim 0.5 1 2"`,
to select any positive fixed radii. The sweep page builds its radius filter from
the completed-run manifest and lists **Free (unconstrained)** plus every actual
radius value present.

## Package for GitHub Pages

After at least one sweep run completes, build a static-site directory with:

```bash
bash demos/package_digits_3d_github_pages.sh
```

The command rebuilds the manifest, copies only completed run JSON files, makes
the sweep selector the site root, renames the trajectory page to `viewer.html`,
and adds `.nojekyll`. Preview `dist/digits-3d-site` locally or publish that
directory through a `gh-pages` branch or GitHub Pages Actions artifact. Three.js
and OrbitControls remain version-pinned CDN modules, so the deployed viewer
requires internet access in the browser.

To initialize the package as a standalone commit and force-push it to the
repository's `gh-pages` branch, run `PUBLISH=true` with the packaging script.
Set `REMOTE_URL` to publish elsewhere. The default packaging command never
pushes, and the generated site remains ignored by the main working tree.
