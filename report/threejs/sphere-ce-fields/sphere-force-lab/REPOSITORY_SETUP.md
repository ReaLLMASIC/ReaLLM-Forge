# Sphere Force Lab — standalone setup

Package version: **9.1.0**. See `VERSION.txt` for source provenance.
This is a browser-only React + Vite app. It needs no account, API key, database,
server application, or platform-specific deployment integration.

## Install and run

Requirements: Node.js **22.13.0 or newer** and pnpm **11.19.0**.

```sh
npm install --global pnpm@11.19.0
cd sphere-force-lab
pnpm install --frozen-lockfile
pnpm dev
```

Open **http://localhost:5173**. Installing packages requires an internet
connection. All model training runs locally in the browser using TensorFlow.js
CPU. No experiment data is sent to a server. Runs and timeline history are held
in memory and disappear on reload.

For access from another device on your local network:

```sh
pnpm dev --host 0.0.0.0
```

Open the printed network URL on the other device. Use this only on a trusted
network; the app has no authentication.

## Build and check

```sh
pnpm test
pnpm build
pnpm preview
```

`pnpm test` runs the quantization, architecture, dataset, and fixed/free-radius checks.
`pnpm build` runs TypeScript checking and produces static files in `dist/`.
`pnpm preview` serves the build at **http://localhost:4173** for local review.
`pnpm start` is an alias for the same local preview.

For deployment, serve the contents of `dist/` from any static HTTP(S) host.
Assets use relative paths so the build also works in a repository subdirectory.
Use a web server rather than double-clicking `index.html`.

## Add to a repository

Extract the single `sphere-force-lab/` directory into your repository. Keep it
as a subdirectory, or copy its contents (including the small configuration
files) to your desired project root. Run package commands in that directory.

This package includes app source, the five UI components it uses, styling,
verification scripts, setup documentation, and a dependency lockfile.
Installed dependencies, generated builds, caches, Git history, and saved
training sessions are not included.

Version 9.1.0 adds free-radius tied rows, mode-aware force views and norm
diagnostics, and explicit optimizer/decay settings. The fixed-radius default is
retained. This remains a standalone static React app with no server-side
authentication, database examples, or deployment integration. See README.md for
the control workflow and compatibility table.

## Main files

- `src/main.tsx`: React entry point
- `src/styles.css`: styling and responsive layout
- `components/sphere-force-lab.tsx`: controls and replay timeline
- `components/sphere-scene.tsx`: Three.js view
- `components/canvas-sphere-fallback.tsx`: compatibility renderer
- `lib/simulator.ts`: model, optimizer, force calculations, and snapshots
- `lib/dataset.ts`: cached direct cycle and seeded Markov generation
- `lib/architecture.ts`: model configuration and limits
- `lib/target-schedule.ts`: inclusion, removal, restoration, and duty cycles
- `scripts/verify-*.mjs`: numerical and behavior checks

See `README.md` for the experiment's definitions and limitations.
