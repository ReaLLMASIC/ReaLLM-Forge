# Changelog

## 1.3.0

- Added independent A/B/C alias-letter and token-ID label visibility controls, with `T` and `I` presenter hotkeys. Aliases and resultant labels remain available when token details are hidden.
- Added versioned JSON settings export/import covering model fields, selected tokens, arithmetic resultants, projection controls, appearance, and camera pose.
- Added guarded pending-workspace restoration: settings never auto-download a model, and saved token selections are restored only after the matching model/revision/vector source is active.
- Added bounded schema validation and safe fallbacks for imported settings files.

## 1.2.0

- Added independent node-label and edge-angle-label size controls.
- Replaced per-label WebGL textures with a high-DPI screen-space label canvas, removing the previous fixed node/edge label display limits.
- Reworked thick-edge animation to update persistent interleaved buffers in place and derive labels from the same validated edge set.
- Added active-tokenizer text import with occurrence-level inspection, individual add, add-all, and replace-set actions.
- Added full-dimensional `slerp(A, B, t)` resultants with deterministic near-antipodal handling and interpolated vector magnitude.
- Added `mean(...)`, `avg(...)`, and `average(...)` helpers to the safe vector-expression language.
