# Sphere Force Lab

Sphere Force Lab is an interactive, CPU-only experiment for inspecting the native 3D geometry of tied token rows in a tiny causal Transformer. It is packaged as a self-contained Next.js app so the whole `sphere-force-lab/` directory can be placed under another repository.

## What the experiment runs

- Model dimension `d = 3`, with every active tied WTE/LM-head row reprojected to radius `sqrt(3)` after each optimizer step.
- One pre-RMSNorm causal self-attention block, one head, and a width-12 GELU MLP.
- A balanced next-token dataset containing all cyclic shifts of `0123456789`.
- AdamW and RMSProp optimizers.
- Optional untargeted rows `a`–`j`, inserted at chosen sphere locations. These rows join the softmax denominator but never appear in inputs or targets.
- Complete iteration history for positions, gradients, tangent forces, optimizer displacements, hidden states, loss, and diagnostics.
- A Three.js view with a Canvas fallback and a responsive phone/tablet layout.

The model and visualization run entirely in the browser. TensorFlow.js is explicitly set to its CPU backend; there is no server-side model or dataset service.

## Surface-force convention

For an active row `w`, ambient cross-entropy gradient `g`, and radius `R`, the displayed constrained force is

```text
F_tangent = -(I - w w^T / R^2) g.
```

The probe heatmap freezes the selected training frame and evaluates a counterfactual new untargeted row `u`:

```text
p_u(s) = sigmoid(u^T h_s - log Z_s)
g(u)   = mean_s p_u(s) h_s
F(u)   = -(I - u u^T / R^2) g(u)
```

This is an instantaneous field at the selected iteration, not a forecast of a row's full future trajectory.

## Run locally

Requirements: Node.js 22.13 or newer and pnpm 11.

```bash
corepack enable
pnpm install
pnpm dev
```

Open `http://localhost:3000`.

Production check:

```bash
pnpm lint
pnpm build
pnpm start
```

## Put it inside another repository

Copy this entire directory beneath the repository root, for example:

```text
your-repository/
├── sphere-force-lab/
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── public/
│   └── package.json
└── ...
```

The folder has no dependency on its parent repository. If the parent is already a pnpm workspace, add `sphere-force-lab` to the root workspace packages and use the root lockfile; otherwise keep the included `pnpm-lock.yaml` and run commands from this directory.

## Important implementation files

- `lib/simulator.ts` — one-layer Transformer, optimizers, projection, snapshots, and analytic probe field.
- `components/sphere-force-lab.tsx` — experiment controls, timeline, metrics, and responsive layout.
- `components/sphere-scene.tsx` — Three.js sphere, rows, arrows, trails, and heatmap.
- `components/canvas-sphere-fallback.tsx` — compatibility renderer when WebGL is unavailable.
- `app/globals.css` — visual system and mobile/tablet breakpoints.

## Repository hygiene

Build output, dependencies, environment files, and local caches are excluded. No credentials, deployment IDs, hosted-site metadata, database scaffolding, or generated results are included.

## Licensing

No project-level license has been selected in this export. Choose and add one before publishing if you want to grant reuse rights. The retained vendored stylesheet includes its original license alongside it in `vendor/`.
