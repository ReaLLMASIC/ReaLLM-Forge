# Repository quick start

This archive contains the complete Sphere Force Lab source for published version 9,
including the Markov transition-matrix dataset editor and fast direct-cycle path.

Source commit: `6f43da1651d25d0f647e947b17187d456f7d893b`  
Exported: 2026-09-24

## Extract and run

Use Node.js 22.13.0 or newer and pnpm 11.25.0 (the version pinned in
`package.json`). Install pnpm if needed:

```sh
npm install --global pnpm@11.25.0
```

Extract the ZIP, then run these commands from the extracted project directory:

```sh
cd sphere-force-lab
pnpm install --frozen-lockfile
pnpm dev
```

Open http://localhost:5173. Dependency installation requires an internet
connection. The experiment runs in the browser; no API key, database, or
backend training service is required. Training runs and timeline history remain
in browser memory and are cleared on reload.

## Include in your repository

Copy the entire extracted `sphere-force-lab` directory into your repository,
or copy its contents to your desired project root. Include the hidden project
configuration files when copying: the build imports `.openai/hosting.json`.
Run the commands above from that project directory. The archive includes the
lockfile, dependency policy, build helpers, documentation, and verification
scripts. It excludes Git history, installed dependencies, build output, caches,
and local runtime state. It does not include saved browser training sessions.

The source files are unchanged from the published commit; this guide is the
only added file. The ZIP has a single top-level `sphere-force-lab/` directory.

## Checks and build

```sh
pnpm exec tsc --noEmit --incremental false
node scripts/verify-qat.mjs
node scripts/verify-architecture.mjs
node scripts/verify-dataset.mjs
pnpm build
```

The verification scripts cover quantization, removal/restoration schedules,
hidden-state means, model architecture, and Markov dataset semantics. See
`README.md` for the experiment's mathematical definitions and limitations.

The framework uses Vinext/Vite with React and produces a Cloudflare Workers
build. A clean extraction selects its portable development profile automatically.
`pnpm start` serves the built Worker locally; use its printed URL.

## Hosting configuration

The included `.openai/hosting.json` identifies the existing Sphere Force Lab
Site and declares no database or object-storage bindings. It contains no access
credential. Keep it for source fidelity and local builds. To deploy a separate
Site, configure that destination with its own project ID before publishing;
other hosting providers require their corresponding deployment configuration.

## Source map

- `app/`: page entry point and global styling
- `components/`: controls, timeline, Three.js scene, and Canvas fallback
- `lib/simulator.ts`: training, optimizer, snapshots, and force diagnostics
- `lib/dataset.ts`: direct-cycle caching and Markov generation
- `lib/architecture.ts`: model configuration and limits
- `lib/target-schedule.ts`: removal, restoration, and duty-cycle membership
- `lib/hidden-means.ts`: target-conditioned hidden-state averages
- `scripts/verify-*.mjs`: numerical and behavior checks
